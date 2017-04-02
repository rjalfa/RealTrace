#ifndef __STRUCTURES_H
#define __STRUCTURES_H
class Triangle;
struct Ray
{
	float3 origin;
	float3 direction;
	int has_intersected;
	Triangle* intersected;
	float t;
};

struct LightSource
{
	float3 position;
	float3 color;
};

class Triangle
{
	public:
		float3 vertexA;
		float3 vertexB;
		float3 vertexC;
		float3 color;
		__host__ __device__ float3 get_normal();
		__host__ __device__ bool intersect(Ray *r);
};

__host__ __device__ float3 get_light_color(float3 point, float3 normal, LightSource* l, Triangle* t, float3 viewVector);

//Structures
__host__ __device__ bool operator==(const float3& v1, const float3& v2);
__host__ __device__ bool operator!=(const float3& v1, const float3& v2);
__host__ __device__ float3 operator+(const float3& v1, const float3& v2);
__host__ __device__ float3 operator-(const float3& v1, const float3& v2);
__host__ __device__ float3 operator-(const float3& v);
__host__ __device__ float3 operator/(const float3& v, float scalar);
__host__ __device__ float3 operator*(const float3& v, float scalar);
__host__ __device__ float3 operator*(float scalar, const float3& v);
__host__ __device__ float3 operator*(const float3& v, const float3& v1);
__host__ __device__ float3 reflect(const float3& I, const float3& N);
__host__ __device__ float squaredlength(const float3& f);
__host__ __device__ float length(const float3& f);
__host__ __device__ float3 normalize(const float3& f);
__host__ __device__ float3 unitVector(const float3& v);
__host__ __device__ float3 crossProduct(const float3& v1, const float3& v2);
__host__ __device__ float distance(const float3& v1, const float3& v2);
__host__ __device__ float dotProduct(const float3& v1, const float3& v2);
__host__ __device__ float tripleProduct(const float3& v1,const float3& v2,const float3& v3);
__host__ __device__ float3 get_point(Ray* r, float t);

#endif
