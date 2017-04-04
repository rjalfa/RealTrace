#include "kernel.h"
#include "structures.h"
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/scan.h>
#include "camera.h"
#include "helper_cuda.h"
#include "cuda_profiler_api.h"
#define TX 32
#define TY 32
#define AMBIENT_COLOR make_float3(0.8083, 1, 1)
#define KR 0.4

__device__
unsigned char clip(float x){ return x > 255 ? 255 : (x < 0 ? 0 : x); }

UniformGrid * d_uniform_grid;

// kernel function to compute decay and shading
__device__ void get_color_from_float3(float3 color, uchar4* cell)
{
  cell->x = clip(color.x*255);
  cell->y = clip(color.y*255);
  cell->z = clip(color.z*255);
  cell->w = 255;
}

//Uniform Grid Intersect
//__device__ void getFirstIntersection(UniformGrid * ug, Ray& r) {
//  ug->intersect(r);
//}

//Global Memory loop intersect
/*
__device__ void intersect(Triangle* triangles, int num_triangles, Ray* r)
{
  for(int i = 0; i < num_triangles; i ++) triangles[i].intersect(r);
}
*/

//Shared Memory Loop Intersect

__device__ void intersect(Triangle* triangles, int num_triangles, Ray* r, UniformGrid * ug)
{
//  __shared__ Triangle localObjects[32];
//  int triangles_to_scan = num_triangles;
//  while(triangles_to_scan > 0)
//  {
//    int x = min(triangles_to_scan,32);
//    if(threadIdx.x == 0 && threadIdx.y < x) localObjects[threadIdx.y] = triangles[threadIdx.y];
//    __syncthreads();
//
//    for(int i = 0; i < x; i ++) localObjects[i].intersect(r);
//    triangles += 32;
//    triangles_to_scan -= 32;
//    __syncthreads();
//  }
	ug->intersect(triangles, *r);
}

__global__ void raytrace(uchar4 *d_out, int w, int h, Camera* camera, Triangle* triangles, int num_triangles, LightSource* l, UniformGrid * ug) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int index = i + j*w; // 1D indexing
  float3 pos = camera->get_position();
  float3 dir = camera->get_ray_direction(i,j);
  Ray r;
  r.origin = pos;
  r.direction = dir;
  r.has_intersected = false;
  r.t = -1;
  r.intersected = 0;
//  Query
   //for(int i = 0; i < num_triangles; i ++) triangles[i].intersect(&r);
//  getFirstIntersection(d_uniform_grid, r);
  //Query
  intersect(triangles,num_triangles,&r, ug);
  
  float3 finalColor;
  
  if(!r.has_intersected) finalColor = AMBIENT_COLOR;
  else 
  {
  	finalColor = (1-KR) * get_light_color( get_point(&r,r.t), r.intersected->get_normal(), l, r.intersected, r.direction);
  	float multiplier = KR;
	float3 pos = get_point(&r,r.t);
	float3 dir = r.direction;
	float3 normal = r.intersected->get_normal();
  	while(multiplier > 1e-4)
  	{
		r.origin = pos + 1e-4;//intersected point;
  		r.direction = reflect(normalize(dir),normalize(normal));//reflect dir;
  		r.has_intersected = false;
  		r.t = -1;
  		r.intersected = 0;
  		intersect(triangles, num_triangles, &r, ug);
		if(!r.has_intersected) {finalColor = finalColor + multiplier * AMBIENT_COLOR; break;}
		else finalColor = finalColor + multiplier * get_light_color( get_point(&r,r.t), r.intersected->get_normal(), l, r.intersected, r.direction);
		pos = get_point(&r,r.t);
		dir = r.direction;
		normal = r.intersected->get_normal();
		multiplier *= KR;
	}
  }
  get_color_from_float3(finalColor, d_out + index);
  //printf("T[%d][%d][%d][%d], c=%d\n", blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y, d_out + index);
  //else get_color_from_float3()
}

int damnCeil(int num, int den) {
  return (num / den) + (num % den != 0);
}

__global__ void get_bounds(float * xmin, float * xmax, float * ymin, float * ymax, float * zmin, float * zmax, Triangle * triangles, int num_triangles) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < num_triangles) {
    triangles[idx].getWorldBound(xmin[idx], xmax[idx], ymin[idx], ymax[idx], zmin[idx], zmax[idx]);
  }
}

__global__ void count_sizes(UniformGrid * ug, Triangle * triangles, int num_triangles) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < num_triangles) {
      float xmin, xmax, ymin, ymax, zmin, zmax;
      triangles[idx].getWorldBound(xmin, xmax, ymin, ymax, zmin, zmax);
      
      int vxmin, vxmax, vymin, vymax, vzmin, vzmax;

      vxmin = ug->posToVoxel(xmin, 0);
      vxmax = ug->posToVoxel(xmax, 0);
      vymin = ug->posToVoxel(ymin, 1);
      vymax = ug->posToVoxel(ymax, 1);
      vzmin = ug->posToVoxel(zmin, 2);
      vzmax = ug->posToVoxel(zmax, 2);

      for(int z = vzmin; z <= vzmax; z++) {
        for(int y = vymin; y <= vymax; y++) {
          for(int x = vxmin; x <= vxmax; x++) {
            int o = ug->offset(x, y, z);
            atomicAdd(&(ug->lower_limit[o]), 1);
          }
        }
      }
  }
}

//__global__ void reserve_space(UniformGrid * ug, int nv) {
//	int idx = blockDim.x * blockIdx.x + threadIdx.x;
//	if(idx < nv) {
//		if(idx > 0) ug->voxels[idx].offset = ug->voxel_sizes[idx - 1];
//		else ug->voxels[idx].offset = 0;
//		if(idx > 0) ug->voxels[idx].max_size = ug->voxel_sizes[idx] - ug->voxel_sizes[idx - 1];
//		else ug->voxels[idx].max_size = ug->voxel_sizes[idx];
//		ug->voxels[idx].curr_size = 0;
////		printf("%d\n", ug->voxels[idx].max_size);
//	}
//}

__global__ void build_grid(UniformGrid * ug, Triangle * triangles, int num_triangles) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < num_triangles) {
      float xmin, xmax, ymin, ymax, zmin, zmax;
      triangles[idx].getWorldBound(xmin, xmax, ymin, ymax, zmin, zmax);
      
      int vxmin, vxmax, vymin, vymax, vzmin, vzmax;

      vxmin = ug->posToVoxel(xmin, 0);
      vxmax = ug->posToVoxel(xmax, 0);
      vymin = ug->posToVoxel(ymin, 1);
      vymax = ug->posToVoxel(ymax, 1);
      vzmin = ug->posToVoxel(zmin, 2);
      vzmax = ug->posToVoxel(zmax, 2);

      for(int z = vzmin; z <= vzmax; z++) {
        for(int y = vymin; y <= vymax; y++) {
          for(int x = vxmin; x <= vxmax; x++) {
            int o = ug->offset(x, y, z);
            	Voxel::addPrimitive(ug, idx, o);
          }
        }
      }
  } 
}

class justMax {
public:
	__host__ __device__
	thrust::tuple<float, float, float> operator()(thrust::tuple<float, float, float> a, thrust::tuple<float, float, float> b) {
		return thrust::make_tuple(max(thrust::get < 0 > (a), thrust::get < 0 > (b)), max(thrust::get < 1 > (a), thrust::get < 1 > (b)), max(thrust::get < 2 > (a), thrust::get < 2 > (b)));
  }
};

class justMin {
public:
	__host__ __device__
	thrust::tuple<float, float, float> operator()(thrust::tuple<float, float, float> a, thrust::tuple<float, float, float> b) {
		return thrust::make_tuple(min(thrust::get < 0 > (a), thrust::get < 0 > (b)), min(thrust::get < 1 > (a), thrust::get < 1 > (b)), min(thrust::get < 2 > (a), thrust::get < 2 > (b)));
  }
};

void buildGrid(int w, int h, Triangle * triangles, int num_triangles) {
  float * xmin, * xmax, * ymin, * ymax, * zmin, * zmax;
  cudaMalloc(&xmin, sizeof(float) * num_triangles);
  cudaMalloc(&xmax, sizeof(float) * num_triangles);
  cudaMalloc(&ymin, sizeof(float) * num_triangles);
  cudaMalloc(&ymax, sizeof(float) * num_triangles);
  cudaMalloc(&zmin, sizeof(float) * num_triangles);
  cudaMalloc(&zmax, sizeof(float) * num_triangles);

  const dim3 blockSize(TX * TY);
  const dim3 gridSizeTriangles(damnCeil(num_triangles, TX * TY));

  get_bounds <<< gridSizeTriangles, blockSize >>> (xmin, xmax, ymin, ymax, zmin, zmax, triangles, num_triangles);

  thrust::tuple <float, float, float> axis_min, axis_max;

  thrust::device_ptr < float > xminptr = thrust::device_pointer_cast(xmin);
  thrust::device_ptr < float > xmaxptr = thrust::device_pointer_cast(xmax);
  thrust::device_ptr < float > yminptr = thrust::device_pointer_cast(ymin);
  thrust::device_ptr < float > ymaxptr = thrust::device_pointer_cast(ymax);
  thrust::device_ptr < float > zminptr = thrust::device_pointer_cast(zmin);
  thrust::device_ptr < float > zmaxptr = thrust::device_pointer_cast(zmax);

  UniformGrid h_uniform_grid;

  h_uniform_grid.bounds.axis_min[0] = thrust::reduce(xminptr, xminptr + num_triangles, 1e36, thrust::minimum<float>());
  h_uniform_grid.bounds.axis_min[1] = thrust::reduce(yminptr, yminptr + num_triangles, 1e36, thrust::minimum<float>());
  h_uniform_grid.bounds.axis_min[2] = thrust::reduce(zminptr, zminptr + num_triangles, 1e36, thrust::minimum<float>());
  h_uniform_grid.bounds.axis_max[0] = thrust::reduce(xmaxptr, xmaxptr + num_triangles, -1e36, thrust::maximum<float>());
  h_uniform_grid.bounds.axis_max[1] = thrust::reduce(ymaxptr, ymaxptr + num_triangles, -1e36, thrust::maximum<float>());
  h_uniform_grid.bounds.axis_max[2] = thrust::reduce(zmaxptr, zmaxptr + num_triangles, -1e36, thrust::maximum<float>());

  h_uniform_grid.initialize(num_triangles);
  
  checkCudaErrors(cudaMalloc(&d_uniform_grid, sizeof(UniformGrid)));
  checkCudaErrors(cudaMemcpy(d_uniform_grid, &h_uniform_grid, sizeof(UniformGrid), cudaMemcpyHostToDevice));

  const dim3 gridSizeVoxels(damnCeil(h_uniform_grid.nv, TX * TY));
  count_sizes <<< gridSizeTriangles, blockSize >>> (d_uniform_grid, triangles, num_triangles);

  checkCudaErrors(cudaMemcpy(&h_uniform_grid, d_uniform_grid, sizeof(UniformGrid), cudaMemcpyDeviceToHost));

  thrust::device_ptr < int > voxel_sizes = thrust::device_pointer_cast(h_uniform_grid.lower_limit);
  int total_space = thrust::reduce(voxel_sizes, voxel_sizes + h_uniform_grid.nv);
  checkCudaErrors(cudaMalloc(&(h_uniform_grid.index_pool), sizeof(int) * total_space));
  thrust::exclusive_scan(voxel_sizes, voxel_sizes + h_uniform_grid.nv, voxel_sizes);

  checkCudaErrors(cudaMemcpy(h_uniform_grid.upper_limit, h_uniform_grid.lower_limit,
		  	  	  sizeof(int) * h_uniform_grid.nv, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_uniform_grid, &h_uniform_grid, sizeof(UniformGrid), cudaMemcpyHostToDevice));

//  reserve_space <<< gridSizeVoxels, blockSize >>> (d_uniform_grid, h_uniform_grid.nv);

  build_grid <<< gridSizeTriangles, blockSize >>> (d_uniform_grid, triangles, num_triangles);

  checkCudaErrors(cudaFree(xmin));
  checkCudaErrors(cudaFree(xmax));
  checkCudaErrors(cudaFree(ymin));
  checkCudaErrors(cudaFree(ymax));
  checkCudaErrors(cudaFree(zmin));
  checkCudaErrors(cudaFree(zmax));
}

void kernelLauncher(uchar4 *d_out, int w, int h, Camera* camera, Triangle* triangles, int num_triangles, LightSource* l) {
  //AMBIENT_COLOR = make_float3()
  const dim3 blockSize(TX, TY);
  const dim3 gridSize = dim3(w/TX,h/TY);
  cudaProfilerStart();
  raytrace<<<gridSize, blockSize>>>(d_out, w, h, camera, triangles, num_triangles, l, d_uniform_grid);
  cudaProfilerStop();
//  exit(0);
 }
