#include "utilities.h"

__host__ __device__ float determinant(float a, float b, float c, float d)
{
	// a b
	// c d
	return a * d - b * c;
}

__host__ __device__ float determinant(float3 col1, float3 col2, float3 col3)
{
	return col1.x * determinant(col2.y, col3.y, col2.z, col3.z) - col1.y * determinant(col2.x, col3.x, col2.z, col3.z) + col1.z * determinant(col2.x, col3.x, col2.y, col3.y);
}

__host__ __device__ float clamp(float a, float b, float c)
{
	if (a < b) return b;
	if (a > c) return c;
	return a;
}