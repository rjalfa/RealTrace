#include "kernel.h"
#include "structures.h"
#include "camera.h"
#define TX 32
#define TY 32
#define AMBIENT_COLOR make_float3(0.8083, 1, 1)

__device__
unsigned char clip(float x){ return x > 255 ? 255 : (x < 0 ? 0 : x); }

// kernel function to compute decay and shading
__device__ void get_color_from_float3(float3 color, uchar4* cell)
{
  cell->x = clip(color.x*255);
  cell->y = clip(color.y*255);
  cell->z = clip(color.z*255);
  cell->w = 255;
}

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
  //Query
  intersect(triangles,num_triangles,&r);

  if(!r.has_intersected) get_color_from_float3(AMBIENT_COLOR,d_out+index);
  else get_color_from_float3(
        get_light_color( get_point(&r,r.t), r.intersected->get_normal(), l, r.intersected, r.direction)
        ,d_out+index);
  //printf("T[%d][%d][%d][%d], c=%d\n", blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y, d_out + index);
  //else get_color_from_float3()
}

void kernelLauncher(uchar4 *d_out, int w, int h, Camera* camera, Triangle* triangles, int num_triangles, LightSource* l) {
  //AMBIENT_COLOR = make_float3()
  const dim3 blockSize(TX, TY);
  const dim3 gridSize = dim3(w/TX,h/TY);
  raytrace<<<gridSize, blockSize>>>(d_out, w, h, camera, triangles, num_triangles, l);
 }
