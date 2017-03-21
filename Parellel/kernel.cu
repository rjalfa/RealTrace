#include "kernel.h"
#include "structures.h"
#define TX 32
#define TY 32
#define AMBIENT_COLOR make_float3(0.1, 0.1, 0.5)

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

__global__ void raytrace(uchar4 *d_out, int w, int h, Ray* rays, Triangle* triangles, int num_triangles, LightSource* l) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int index = i + j*w; // 1D indexing
  Ray r = rays[index];
  r.has_intersected = false;
  r.t = -1;
  r.intersected = 0;
  //Query
  for(int i = 0; i < num_triangles; i ++) triangles[i].intersect(&r);
  if(!r.has_intersected) get_color_from_float3(AMBIENT_COLOR,d_out+index);
  else get_color_from_float3(
        get_light_color( get_point(&r,r.t), r.intersected->get_normal(), l, r.intersected, r.direction)
        ,d_out+index);
  //else get_color_from_float3()
}

void kernelLauncher(uchar4 *d_out, int w, int h, Ray* rays, Triangle* triangles, int num_triangles) {
  //AMBIENT_COLOR = make_float3()
  //const dim3 blockSize(TX, TY);
  //const dim3 gridSize = dim3((w + TX - 1)/TX, (h + TY - 1)/TY);
  //raytrace<<<gridSize, blockSize>>>(d_out, w, h, p, s);
 }