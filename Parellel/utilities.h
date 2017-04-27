#ifndef __UTILITIES_H
#define __UTILITIES_H
#define EPSILON 1e-4f
__host__ __device__ float determinant(float a, float b, float c, float d);
__host__ __device__ float determinant(float3 col1, float3 col2, float3 col3);
__host__ __device__ float clamp(float a, float b, float c);

#endif