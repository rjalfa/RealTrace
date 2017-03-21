#ifndef __UTILITIES_H
#define __UTILITIES_H
#define EPSILON 0.0001
__device__ float determinant(float a,float b,float c,float d);
extern "C" __device__ float determinant(float3 col1, float3 col2, float3 col3);

#endif