#include "kernel.h"
#include "structures.h"
#define TX 32
#define TY 32
#define AMBIENT_COLOR make_float3(0.1, 0.1, 0.5)

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

__device__ void getFirstIntersection(UniformGrid * ug, Ray& r) {
  ug->intersect(r);
}

__global__ void raytrace(uchar4 *d_out, int w, int h, Ray* rays, Triangle* triangles, int num_triangles, LightSource* l) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int index = i + j*w; // 1D indexing
  Ray r = rays[index];
  r.has_intersected = false;
  r.t = -1;
  r.intersected = 0;
  //Query
  // for(int i = 0; i < num_triangles; i ++) triangles[i].intersect(&r);
  getFirstIntersection(d_uniform_grid, r);

  if(!r.has_intersected) get_color_from_float3(AMBIENT_COLOR,d_out+index);
  else get_color_from_float3(
        get_light_color( get_point(&r,r.t), r.intersected->get_normal(), l, r.intersected, r.direction)
        ,d_out+index);
  //else get_color_from_float3()
}

int damnCeil(int num, int den) {
  return (num / den) + (num % den != 0);
}

class justMax() {
public:
  float3 operator(float3 a, float3 b) {
    return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
  }
}

class justMin() {
public:
  float3 operator(float3 a, float3 b) {
    return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
  }
}

__global__ void get_bounds(int * xmin, int * xmax, int * ymin, int * ymax, int * zmin, int * zmax, Triangle * triangles, int num_triangles) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < num_triangles) {
    triangles[idx]->getWorldBounds(xmin[idx], xmax[idx], ymin[idx], ymax[idx], zmin[idx], zmax[idx]);
  }
}

__global__ void reserve_sizes(UniformGrid * ug, Triangle * triangles, int num_triangles) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < num_triangles) {
      int xmin, xmax, ymin, ymax, zmin, zmax;
      triangles[idx]->getWorldBound(xmin, xmax, ymin, ymax, zmin, zmax);
      
      int vxmin, vxmax, vymin, vymax, vzmin, vzmax;

      vxmin = ug->posToVoxel(Vector3D(xmin, ymin, zmin), 0);
      vxmax = ug->posToVoxel(Vector3D(xmax, ymax, zmax), 0);
      vymin = ug->posToVoxel(Vector3D(xmin, ymin, zmin), 1);
      vymax = ug->posToVoxel(Vector3D(xmax, ymax, zmax), 1);
      vzmin = ug->posToVoxel(Vector3D(xmin, ymin, zmin), 2);
      vzmax = ug->posToVoxel(Vector3D(xmax, ymax, zmax), 2);

      for(int z = vzmin; z <= vzmax; z++) {
        for(int y = vymin; y <= vymax; y++) {
          for(int x = vxmin; x <= vxmax; x++) {
            int o = ug->offset(x, y, z);
            atomicAdd(&(ug->voxel_sizes[o]), 1);
          }
        }
      }

      if(idx < ug->nv) {
        ug->voxels[idx].resize(voxel_sizes[idx]);
      }
  }
}

__global__ void build_grid(UniformGrid * ug, Triangle * triangles, int num_triangles) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < num_triangles) {
      int xmin, xmax, ymin, ymax, zmin, zmax;
      triangles[idx]->getWorldBound(xmin, xmax, ymin, ymax, zmin, zmax);
      
      int vxmin, vxmax, vymin, vymax, vzmin, vzmax;

      vxmin = ug->posToVoxel(Vector3D(xmin, ymin, zmin), 0);
      vxmax = ug->posToVoxel(Vector3D(xmax, ymax, zmax), 0);
      vymin = ug->posToVoxel(Vector3D(xmin, ymin, zmin), 1);
      vymax = ug->posToVoxel(Vector3D(xmax, ymax, zmax), 1);
      vzmin = ug->posToVoxel(Vector3D(xmin, ymin, zmin), 2);
      vzmax = ug->posToVoxel(Vector3D(xmax, ymax, zmax), 2);

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

void buildGrid(int w, int h, Triangle * triangles, int num_triangles) {
  int * xmin, * xmax, * ymin, * ymax, * zmin, * zmax;
  cudaMalloc(&xmin, sizeof(int) * num_triangles);
  cudaMalloc(&xmax, sizeof(int) * num_triangles);
  cudaMalloc(&ymin, sizeof(int) * num_triangles);
  cudaMalloc(&ymax, sizeof(int) * num_triangles);
  cudaMalloc(&zmin, sizeof(int) * num_triangles);
  cudaMalloc(&zmax, sizeof(int) * num_triangles);

  const dim3 blockSize(TX * TY);
  const dim3 gridSize(damnCeil(num_triangles, TX * TY));

  get_bounds <<< gridSize, blockSize >>> (xmin, xmax, ymin, ymax, zmin, zmax, triangles, num_triangles);

  float3 axis_min = reduce(make_zip_iterator(make_tuple(xmin, ymin, zmin)), make_zip_iterator(make_tuple(xmin + num_triangles, ymin + num_triangles, zmin + num_triangles)));
  float3 axis_max = reduce(make_zip_iterator(make_tuple(xmax, ymax, zmax)), make_zip_iterator(make_tuple(xmax + num_triangles, ymax + num_triangles, zmax + num_triangles)));

  UniformGrid h_uniform_grid;
  h_uniform_grid->bounds.axis_min[0] = axis_min.x;
  h_uniform_grid->bounds.axis_min[1] = axis_min.y;
  h_uniform_grid->bounds.axis_min[2] = axis_min.z;
  h_uniform_grid->bounds.axis_max[0] = axis_max.x;
  h_uniform_grid->bounds.axis_max[1] = axis_max.y;
  h_uniform_grid->bounds.axis_max[2] = axis_max.z;

  h_uniform_grid.initialize(num_triangles);
  
  cudaMemcpy(d_uniform_grid, h_uniform_grid, sizeof(UniformGrid), cudaMemcpyHostToDevice);

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
 }