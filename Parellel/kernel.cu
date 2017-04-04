#include "kernel.h"
#include "structures.h"
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/scan.h>
#include "camera.h"
#include "helper_cuda.h"
#define TX 32
#define TY 32
#define AMBIENT_COLOR make_float3(0.8083, 1, 1)
#define KR 0.4

__device__
unsigned char clip(float x){ return x > 255 ? 255 : (x < 0 ? 0 : x); }

UniformGrid * d_uniform_grid;
float3* colors = 0;
Ray* d_rays = 0;

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

////////////////////////////////////////////////////////////////////////////
// Ray generation kernel
// Parameters:
// camera = Camera object
// w = width
// h = height
// out_rays = Output rays
// d_out = Output image to be resetted
////////////////////////////////////////////////////////////////////////////
__global__ void createRaysAndResetImage(Camera* camera, int w, int h, Ray* out_rays, uchar4* d_out)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    float3 pos = camera->get_position();
    float3 dir = camera->get_ray_direction(i,j);
    int index = i + j*w; // 1D indexing
    Ray r;
    r.origin = pos;
    r.direction = dir;
    r.has_intersected = false;
    r.t = -1;
    r.intersected = 0;
    out_rays[index] = r;
    d_out[index] = make_uchar4(0,0,0,0);
}

////////////////////////////////////////////////////////////////////////////
// Recursive Ray-tracing Kernel
// Parameters:
// out_color = Global Color Array that stores output from all kernels
// in_coeffs = The coeffs for the current kernel rays. If NULL, assumed all 1's
// w = width
// h = height
// rays = The rays to trace for this kernel
// out_rays_reflect = The rays that emerge from reflection from this kernel, If NULL, assumed end of recursion
// out_rays_refract = The rays that emerge from reflection from this kernel, If NULL, assumed end of recursion
// out_coeffs_reflect = The coeffs for the reflected rays
// out_coeffs_refract = The coeffs for the refracted rays
// triangles = Triangle objects
// num_triangles = Number of triangles in above
// l = LightSource object
// ug = UniformGrid object
////////////////////////////////////////////////////////////////////////////
__global__ void raytrace(float3 *out_color, float* in_coeffs, int w, int h, Ray* rays, Ray* out_rays_reflect, Ray* out_rays_refract, float* out_coeffs_reflect, float* out_coeffs_refract, Triangle* triangles, int num_triangles, LightSource* l, UniformGrid * ug)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int index = i + j * w;

  //Switches
  bool in_coeffs_defined = (in_coeffs != NULL);
  bool can_refract = (out_rays_refract != NULL && out_coeffs_refract != NULL);
  bool can_reflect = (out_rays_reflect != NULL && out_coeffs_reflect != NULL);

  //Get owned ray
  Ray r = rays[index];
  intersect(triangles,num_triangles,&r, ug);
  
  //If only diffuse possible, do one time intersection
  float3 finalColor = make_float3(0,0,0);
  if(!can_reflect && !can_refract)
  {
    if(!r.has_intersected) finalColor = AMBIENT_COLOR;
    else finalColor = get_light_color(get_point(&r,r.t), r.intersected->get_normal(), l, r.intersected, r.direction);
  }
  else
  {
    // Do something
  }

  out_color[index] = finalColor;
};

///////////////////////////////////////////////////////////////////
// Convert to RGBA kernel
// Parameters:
// color = Color array in floats 
// d_out = Output array as RGBA unsigned char
// w = width
// h = height
//////////////////////////////////////////////////////////////////
__global__ void convert_to_rgba(float3 *color, uchar4* d_out, int w, int h)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    int index = i + j*w; // 1D indexing
    get_color_from_float3(color[index],d_out + index);
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
            atomicAdd(&(ug->voxel_sizes[o]), 1);
        }
    }
}
}
}

__global__ void reserve_space(UniformGrid * ug, int nv) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < nv) {
//		ug->voxels[idx].primitives = (int * ) malloc(ug->voxel_sizes[idx] * sizeof(int));
//		if(ug->voxels[idx].primitives == 0) {
//			if(ug->voxel_sizes[idx] != 0)
//				printf("couldn't allocate %d for: %d\n", ug->voxel_sizes[idx], idx);
//		}
		if(idx > 0) ug->voxels[idx].offset = ug->voxel_sizes[idx - 1];
		else ug->voxels[idx].offset = 0;
		if(idx > 0) ug->voxels[idx].max_size = ug->voxel_sizes[idx] - ug->voxel_sizes[idx - 1];
		else ug->voxels[idx].max_size = ug->voxel_sizes[idx];
		ug->voxels[idx].curr_size = 0;
//		printf("%d\n", ug->voxels[idx].max_size);
	}
}

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
            ug->voxels[o].addPrimitive(ug, idx);
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
  checkCudaErrors(cudaMalloc((void**)&colors, sizeof(float3)*w*h));
  checkCudaErrors(cudaMalloc((void**)&d_rays, sizeof(Ray)*w*h));

  float * xmin, * xmax, * ymin, * ymax, * zmin, * zmax;
  cudaMalloc(&xmin, sizeof(float) * num_triangles);
  cudaMalloc(&xmax, sizeof(float) * num_triangles);
  cudaMalloc(&ymin, sizeof(float) * num_triangles);
  cudaMalloc(&ymax, sizeof(float) * num_triangles);
  cudaMalloc(&zmin, sizeof(float) * num_triangles);
  cudaMalloc(&zmax, sizeof(float) * num_triangles);

//  printf("%x\n", xmin);
//  std::cerr << "allocation done" << endl;
  const dim3 blockSize(TX * TY);
  const dim3 gridSizeTriangles(damnCeil(num_triangles, TX * TY));

  get_bounds <<< gridSizeTriangles, blockSize >>> (xmin, xmax, ymin, ymax, zmin, zmax, triangles, num_triangles);
//  printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
//  cudaDeviceSynchronize();
//  std::cerr << "kernel done" << endl;

  thrust::tuple <float, float, float> axis_min, axis_max;

//  thrust::device_ptr<float> cptr = thrust::device_pointer_cast(xmin);
//  float mn = thrust::reduce(cptr, cptr + num_triangles, 1e36, thrust::minimum<float>());
//  cudaDeviceSynchronize();
//  std::cerr << mn << endl;

  thrust::device_ptr < float > xminptr = thrust::device_pointer_cast(xmin);
  thrust::device_ptr < float > xmaxptr = thrust::device_pointer_cast(xmax);
  thrust::device_ptr < float > yminptr = thrust::device_pointer_cast(ymin);
  thrust::device_ptr < float > ymaxptr = thrust::device_pointer_cast(ymax);
  thrust::device_ptr < float > zminptr = thrust::device_pointer_cast(zmin);
  thrust::device_ptr < float > zmaxptr = thrust::device_pointer_cast(zmax);

//  thrust::reduce(thrust::make_zip_iterator(thrust::make_tuple(xminptr), xminptr + num_triangles, 1e36, thrust::minimum<float>());
//  thrust::make_zip_iterator(thrust::make_tuple(xminptr, yminptr, zminptr));
//  axis_min = thrust::reduce(thrust::make_zip_iterator(thrust::make_tuple(xminptr, yminptr, zminptr)),
//		  	 thrust::make_zip_iterator(thrust::make_tuple(xminptr + num_triangles, yminptr + num_triangles, zminptr + num_triangles)),
//		  	 thrust::make_tuple(1e36, 1e36, 1e36), justMin());
//  cudaDeviceSynchronize();
//  std::cerr << thrust::get < 0 > (axis_min) << " " << thrust::get < 1 > (axis_min) << " " << thrust::get < 2 > (axis_min) << endl;

//  axis_max = thrust::reduce(thrust::make_zip_iterator(thrust::make_tuple(xmaxptr, ymaxptr, zmaxptr)),
//		  	 thrust::make_zip_iterator(thrust::make_tuple(xmaxptr + num_triangles, ymaxptr + num_triangles, zmaxptr + num_triangles)),
//		  	 thrust::make_tuple(-1e36, -1e36, -1e36), justMax());
//
//  cudaDeviceSynchronize();
//  std::cerr << thrust::get < 0 > (axis_max) << " " << thrust::get < 1 > (axis_max) << " " << thrust::get < 2 > (axis_max) << endl;

  UniformGrid h_uniform_grid;
//  h_uniform_grid.bounds.axis_min[0] = thrust::get < 0 > (axis_min);
//  h_uniform_grid.bounds.axis_min[1] = thrust::get < 1 > (axis_min);
//  h_uniform_grid.bounds.axis_min[2] = thrust::get < 2 > (axis_min);
//  h_uniform_grid.bounds.axis_max[0] = thrust::get < 0 > (axis_max);
//  h_uniform_grid.bounds.axis_max[1] = thrust::get < 1 > (axis_max);
//  h_uniform_grid.bounds.axis_max[2] = thrust::get < 2 > (axis_max);

  h_uniform_grid.bounds.axis_min[0] = thrust::reduce(xminptr, xminptr + num_triangles, 1e36, thrust::minimum<float>());
  h_uniform_grid.bounds.axis_min[1] = thrust::reduce(yminptr, yminptr + num_triangles, 1e36, thrust::minimum<float>());
  h_uniform_grid.bounds.axis_min[2] = thrust::reduce(zminptr, zminptr + num_triangles, 1e36, thrust::minimum<float>());
  h_uniform_grid.bounds.axis_max[0] = thrust::reduce(xmaxptr, xmaxptr + num_triangles, -1e36, thrust::maximum<float>());
  h_uniform_grid.bounds.axis_max[1] = thrust::reduce(ymaxptr, ymaxptr + num_triangles, -1e36, thrust::maximum<float>());
  h_uniform_grid.bounds.axis_max[2] = thrust::reduce(zmaxptr, zmaxptr + num_triangles, -1e36, thrust::maximum<float>());

//  cudaDeviceSynchronize();
//  std::cerr << h_uniform_grid.bounds.axis_min[0] << " " << h_uniform_grid.bounds.axis_min[1] << " " << h_uniform_grid.bounds.axis_min[2] << endl;

  h_uniform_grid.initialize(num_triangles);
  
  checkCudaErrors(cudaMalloc(&d_uniform_grid, sizeof(UniformGrid)));
  checkCudaErrors(cudaMemcpy(d_uniform_grid, &h_uniform_grid, sizeof(UniformGrid), cudaMemcpyHostToDevice));

  const dim3 gridSizeVoxels(damnCeil(h_uniform_grid.nv, TX * TY));
  count_sizes <<< gridSizeTriangles, blockSize >>> (d_uniform_grid, triangles, num_triangles);

  checkCudaErrors(cudaMemcpy(&h_uniform_grid, d_uniform_grid, sizeof(UniformGrid), cudaMemcpyDeviceToHost));

  thrust::device_ptr < int > voxel_sizes = thrust::device_pointer_cast(h_uniform_grid.voxel_sizes);
  int total_space = thrust::reduce(voxel_sizes, voxel_sizes + h_uniform_grid.nv);
  checkCudaErrors(cudaMalloc(&(h_uniform_grid.index_pool), sizeof(int) * total_space));
  thrust::inclusive_scan(voxel_sizes, voxel_sizes + h_uniform_grid.nv, voxel_sizes);

  checkCudaErrors(cudaMemcpy(d_uniform_grid, &h_uniform_grid, sizeof(UniformGrid), cudaMemcpyHostToDevice));

  reserve_space <<< gridSizeVoxels, blockSize >>> (d_uniform_grid, h_uniform_grid.nv);
//  cudaDeviceSynchronize();

//  printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));

  build_grid <<< gridSizeTriangles, blockSize >>> (d_uniform_grid, triangles, num_triangles);
//  printf("Error: %s\n", cudaGetErrorString(cudaGetLastError()));
//  printf("%x\n", xmin);
  checkCudaErrors(cudaFree(xmin));
  checkCudaErrors(cudaFree(xmax));
  checkCudaErrors(cudaFree(ymin));
  checkCudaErrors(cudaFree(ymax));
  checkCudaErrors(cudaFree(zmin));
  checkCudaErrors(cudaFree(zmax));

//  checkCudaErrors(cudaMemcpy(&h_uniform_grid, d_uniform_grid, sizeof(UniformGrid), cudaMemcpyDeviceToHost));
//  int voxel_sizes[h_uniform_grid.nv];
//  checkCudaErrors(cudaMemcpy(voxel_sizes, h_uniform_grid.voxel_sizes, sizeof(voxel_sizes), cudaMemcpyDeviceToHost));
//  for(int i = 0; i < h_uniform_grid.nv; i++) {
//	  std::cout << voxel_sizes[i] << endl;
//  }
}

void kernelLauncher(uchar4 *d_out, int w, int h, Camera* camera, Triangle* triangles, int num_triangles, LightSource* l) {
  const dim3 blockSize(TX, TY);
  const dim3 gridSize = dim3(w/TX,h/TY);
  
  //Start Procedure
  createRaysAndResetImage<<<gridSize, blockSize>>>(camera, w, h, d_rays, d_out);
  
  //Karlo Ray trace 1000 baar yahaan
  raytrace<<<gridSize, blockSize>>>(colors, NULL, w, h, d_rays, NULL, NULL, NULL, NULL, triangles, num_triangles, l, d_uniform_grid);
  //...
  
  //Final Output Array
  convert_to_rgba<<<gridSize, blockSize>>>(colors, d_out, w, h);
}   
