#ifndef __UTILITIES_H
#define __UTILITIES_H
#define EPSILON 0.0001
__host__ __device__ float determinant(float a,float b,float c,float d);
__host__ __device__ float determinant(float3 col1, float3 col2, float3 col3);

#endif