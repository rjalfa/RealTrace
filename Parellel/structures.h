#ifndef __STRUCTURES_H
#define __STRUCTURES_H

class Triangle
{
	public:
		float3 vertex1;
		float3 vertex2;
		float3 vertex3;
		__device__ __host__ float3 get_normal();
};

struct Ray
{
	float3 origin;
	float3 direction;
};

struct LightSource
{
	float3 color;
};

__device__ __host__ bool intersect(Triangle* t, Ray* r);

__device__ __host__ float determinant();

__device__ float3 get_light_color(float3 point, float3 normal, LightSource* l);

//Structures
__device__ bool operator==(const float3& v1, const float3& v2);
__device__ bool operator!=(const float3& v1, const float3& v2);
__device__ float3 operator+(const float3& v1, const float3& v2);
__device__ float3 operator-(const float3& v1, const float3& v2);
__device__ float3 operator/(const float3& v, float scalar);
__device__ float3 operator*(const float3& v, float scalar);
__device__ float3 operator*(float scalar, const float3& v);
__device__ float3 operator*(const float3& v, const float3& v1);
__device__ float squaredlength(const float3& f);
__device__ float length(const float3& f);
__device__ float3 normalize(const float3& f);
__device__ float3 unitVector(const float3& v);
__device__ float3 crossProduct(const float3& v1, const float3& v2);
__device__ float dotProduct(const float3& v1, const float3& v2);
__device__ float tripleProduct(const float3& v1,const float3& v2,const float3& v3);

#endif