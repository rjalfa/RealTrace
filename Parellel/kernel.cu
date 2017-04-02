#include "kernel.h"
#include "structures.h"
<<<<<<< HEAD
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

=======
#include "camera.h"
>>>>>>> master
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

<<<<<<< HEAD
//__device__ void getFirstIntersection(UniformGrid * ug, Ray& r) {
//  ug->intersect(r);
//}

__global__ void raytrace(uchar4 *d_out, int w, int h, Ray* rays, Triangle* triangles, int num_triangles, LightSource* l) {
=======
//Global Memory loop intersect
/*
__device__ void intersect(Triangle* triangles, int num_triangles, Ray* r)
{
  for(int i = 0; i < num_triangles; i ++) triangles[i].intersect(r);
}
*/

//Shared Memory Loop Intersect

__device__ void intersect(Triangle* triangles, int num_triangles, Ray* r)
{
  __shared__ Triangle localObjects[32];
  int triangles_to_scan = num_triangles;
  while(triangles_to_scan > 0)
  {
    int x = min(triangles_to_scan,32);
    if(threadIdx.x == 0 && threadIdx.y < x) localObjects[threadIdx.y] = triangles[threadIdx.y];
    __syncthreads();

    for(int i = 0; i < x; i ++) localObjects[i].intersect(r);
    triangles += 32; 
    triangles_to_scan -= 32;
    __syncthreads();
  }
}


__global__ void raytrace(uchar4 *d_out, int w, int h, Camera* camera, Triangle* triangles, int num_triangles, LightSource* l) {
>>>>>>> master
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
<<<<<<< HEAD
//  Query
   for(int i = 0; i < num_triangles; i ++) triangles[i].intersect(&r);
//  getFirstIntersection(d_uniform_grid, r);

  if(!r.has_intersected) get_color_from_float3(AMBIENT_COLOR,d_out+index);
  else get_color_from_float3(
        get_light_color( get_point(&r,r.t), r.intersected->get_normal(), l, r.intersected, r.direction)
        ,d_out+index);
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

__global__ void reserve_sizes(UniformGrid * ug, Triangle * triangles, int num_triangles) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < num_triangles) {
      float xmin, xmax, ymin, ymax, zmin, zmax;
      triangles[idx].getWorldBound(xmin, xmax, ymin, ymax, zmin, zmax);
      
      int vxmin, vxmax, vymin, vymax, vzmin, vzmax;

      vxmin = ug->posToVoxel(make_float3(xmin, ymin, zmin), 0);
      vxmax = ug->posToVoxel(make_float3(xmax, ymax, zmax), 0);
      vymin = ug->posToVoxel(make_float3(xmin, ymin, zmin), 1);
      vymax = ug->posToVoxel(make_float3(xmax, ymax, zmax), 1);
      vzmin = ug->posToVoxel(make_float3(xmin, ymin, zmin), 2);
      vzmax = ug->posToVoxel(make_float3(xmax, ymax, zmax), 2);

      for(int z = vzmin; z <= vzmax; z++) {
        for(int y = vymin; y <= vymax; y++) {
          for(int x = vxmin; x <= vxmax; x++) {
            int o = ug->offset(x, y, z);
            atomicAdd(&(ug->voxel_sizes[o]), 1);
          }
        }
      }

      if(idx < ug->nv) {
        ug->voxels[idx].primitives = (int * ) malloc(ug->voxel_sizes[idx] * sizeof(int));
        ug->voxels[idx].max_size = ug->voxel_sizes[idx];
      }
  }
}

__global__ void build_grid(UniformGrid * ug, Triangle * triangles, int num_triangles) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < num_triangles) {
      float xmin, xmax, ymin, ymax, zmin, zmax;
      triangles[idx].getWorldBound(xmin, xmax, ymin, ymax, zmin, zmax);
      
      int vxmin, vxmax, vymin, vymax, vzmin, vzmax;

      vxmin = ug->posToVoxel(make_float3(xmin, ymin, zmin), 0);
      vxmax = ug->posToVoxel(make_float3(xmax, ymax, zmax), 0);
      vymin = ug->posToVoxel(make_float3(xmin, ymin, zmin), 1);
      vymax = ug->posToVoxel(make_float3(xmax, ymax, zmax), 1);
      vzmin = ug->posToVoxel(make_float3(xmin, ymin, zmin), 2);
      vzmax = ug->posToVoxel(make_float3(xmax, ymax, zmax), 2);

      for(int z = vzmin; z <= vzmax; z++) {
        for(int y = vymin; y <= vymax; y++) {
          for(int x = vxmin; x <= vxmax; x++) {
            int o = ug->offset(x, y, z);
            int req_idx = atomicAdd(&(ug->voxels[o].curr_size), 1);
            ug->voxels[o].addPrimitive(req_idx, idx);
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
  const dim3 gridSize(damnCeil(num_triangles, TX * TY));

  get_bounds <<< gridSize, blockSize >>> (xmin, xmax, ymin, ymax, zmin, zmax, triangles, num_triangles);

  thrust::tuple <float, float, float> axis_min, axis_max;
  axis_min = thrust::reduce(thrust::make_zip_iterator(thrust::make_tuple(xmin, ymin, zmin)),
		  	 thrust::make_zip_iterator(thrust::make_tuple(xmin + num_triangles, ymin + num_triangles, zmin + num_triangles)),
		  	 thrust::make_tuple(xmin[0], ymin[0], zmin[0]), justMin());
  axis_max = thrust::reduce(thrust::make_zip_iterator(thrust::make_tuple(xmax, ymax, zmax)),
		  	 thrust::make_zip_iterator(thrust::make_tuple(xmax + num_triangles, ymax + num_triangles, zmax + num_triangles)),
		  	 thrust::make_tuple(xmax[0], ymax[0], zmax[0]), justMax());

  UniformGrid h_uniform_grid;
  h_uniform_grid.bounds.axis_min[0] = thrust::get < 0 > (axis_min);
  h_uniform_grid.bounds.axis_min[1] = thrust::get < 1 > (axis_min);
  h_uniform_grid.bounds.axis_min[2] = thrust::get < 2 > (axis_min);
  h_uniform_grid.bounds.axis_max[0] = thrust::get < 0 > (axis_max);
  h_uniform_grid.bounds.axis_max[1] = thrust::get < 1 > (axis_max);
  h_uniform_grid.bounds.axis_max[2] = thrust::get < 2 > (axis_max);

  h_uniform_grid.initialize(num_triangles);
  
  cudaMemcpy(d_uniform_grid, &h_uniform_grid, sizeof(UniformGrid), cudaMemcpyHostToDevice);

  reserve_sizes <<< gridSize, blockSize >>> (d_uniform_grid, triangles, num_triangles);

  build_grid <<< gridSize, blockSize >>> (d_uniform_grid, triangles, num_triangles);

  cudaFree(xmin);
  cudaFree(xmax);
  cudaFree(ymin);
  cudaFree(ymax);
  cudaFree(zmin);
  cudaFree(zmax);
}

void kernelLauncher(uchar4 *d_out, int w, int h, Ray* rays, Triangle* triangles, int num_triangles, LightSource* l) {
  //AMBIENT_COLOR = make_float3()
  const dim3 blockSize(TX, TY);
  const dim3 gridSize = dim3(w/TX,h/TY);
  
  raytrace<<<gridSize, blockSize>>>(d_out, w, h, rays, triangles, num_triangles, l);

  //Query
  intersect(triangles,num_triangles,&r);
  
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
  		intersect(triangles, num_triangles, &r);
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

void kernelLauncher(uchar4 *d_out, int w, int h, Camera* camera, Triangle* triangles, int num_triangles, LightSource* l) {
  //AMBIENT_COLOR = make_float3()
  const dim3 blockSize(TX, TY);
  const dim3 gridSize = dim3(w/TX,h/TY);
  raytrace<<<gridSize, blockSize>>>(d_out, w, h, camera, triangles, num_triangles, l);

 }
