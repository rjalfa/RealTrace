#ifndef __STRUCTURES_H
#define __STRUCTURES_H
#include <thrust/device_vector.h>
#include <vector>
#define REFRACTIVE 0
#define REFLECTIVE 2
#define DIFFUSE 1
using namespace std;

class Triangle;

//Structures
__host__ __device__ __forceinline__ bool operator==(const float3& v1, const float3& v2)
{
	return (v1.x == v2.x) && (v1.y == v2.y) && (v1.z == v2.z);
}

__host__ __device__ __forceinline__ bool operator!=(const float3& v1, const float3& v2)
{
	return !(v1 == v2);
}

__host__ __device__ __forceinline__ void operator+=(float3& v1, const float3& v2)
{
	v1.x += v2.x;
	v1.y += v2.y;
	v1.z += v2.z;
}

__host__ __device__ __forceinline__ void operator*=(float3& v1, const float& v2)
{
	v1.x *= v2;
	v1.y *= v2;
	v1.z *= v2;
}

__host__ __device__ __forceinline__ float3 operator+(const float3& v1, const float3& v2)
{
	return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__host__ __device__ __forceinline__ float3 operator+(const float3& v1, const double& v2)
{
	return make_float3(v1.x + v2, v1.y + v2, v1.z + v2);
}

__host__ __device__ __forceinline__ float3 operator-(const float3& v1, const float3& v2)
{
	return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__host__ __device__ __forceinline__ float3 operator/(const float3& v, float scalar)
{
	return make_float3(v.x / scalar, v.y / scalar, v.z / scalar);
}

__host__ __device__ __forceinline__ float3 operator*(const float3& v, float scalar)
{
	return make_float3(v.x * scalar, v.y * scalar, v.z * scalar);
}

//Unary
__host__ __device__ __forceinline__ float3 operator-(const float3& v)
{
	return make_float3(-v.x, -v.y, -v.z);
}

__host__ __device__ __forceinline__ float3 operator*(float scalar, const float3& v)
{
	return v * scalar;
}

__host__ __device__ __forceinline__ float3 operator*(const float3& v, const float3& v1)
{
	return make_float3(v.x * v1.x, v.y * v1.y, v.z * v1.z);
}

__device__ float3 reflect(const float3& I, const float3& N);
__device__ float squaredlength(const float3& f);
__device__ float length(const float3& f);
__device__ float3 normalize(const float3& f);
__device__ float3 unitVector(const float3& v);
__host__ __device__ float3 crossProduct(const float3& v1, const float3& v2);
__device__ float distance(const float3& v1, const float3& v2);
__device__ __forceinline__ float dotProduct(const float3& v1, const float3& v2)
{	
	return __fmaf_rz(v1.x, v2.x, __fmaf_rz(v1.y , v2.y , v1.z * v2.z));
}
__device__ float tripleProduct(const float3& v1, const float3& v2, const float3& v3);

struct Ray
{
	float3 origin;
	float3 direction;
	int has_intersected;
	Triangle* intersected;
	float t;
	__host__ __device__
	void strictSetParameter(float para) {
		t = para;
	}
	__host__ __device__
	float3 getPosition() {
		if (has_intersected) return origin + t * direction;
		else return make_float3(0, 0, 0);
	}
	__host__ __device__ Ray(float3 origin_p, float3 direction_p)
	{
		origin = origin_p;
		direction = direction_p;
		has_intersected = false;
		intersected = 0;
		t = 1e12;
	}
};

struct LightSource
{
	float3 position;
	float3 color;
};

class BBox {
public:
	float axis_min[3], axis_max[3];
	__host__ __device__
	BBox() {
		// GPU isn't as precise, might require changing this
//		axis_min[0] = axis_min[1] = axis_min[2] = std::numeric_limits < float >::max();
//		axis_max[0] = axis_max[1] = axis_max[2] = std::numeric_limits < float >::min();
		axis_min[0] = axis_min[1] = axis_min[2] = 1e36;
		axis_max[0] = axis_max[1] = axis_max[2] = -1e36;
	}
};

class Triangle
{
public:
	float3 vertexA;
	float3 vertexB;
	float3 vertexC;
	float3 color;
	float3 normal;
	bool normal_c;
	Triangle() { normal_c = false; }
	int type_of_material;
	__device__ float3 get_normal();
	__host__ __device__ bool intersect(Ray *r);
	__host__ __device__ BBox getWorldBound();
	__host__ __device__ void getWorldBound(float& xmin, float& xmax, float& ymin, float& ymax, float& zmin, float& zmax);
	__host__ __device__ float3 getVertex(int vno) {
		if (vno == 0) return vertexA;
		else if (vno == 1) return vertexB;
		else return vertexC;
	}
};

class UniformGrid;

class Voxel {
public:
	__device__ static void addPrimitive(UniformGrid * ug, int i, int idx);
	__device__ static bool intersect(UniformGrid * ug, Triangle * triangles, Ray& ray, int idx);
};

class UniformGrid {
	float delta[3];
	int nVoxels[3];
	float voxelsPerUnitDist;
	float width[3], invWidth[3];

	__host__ float findVoxelsPerUnitDist(float delta[], int num);

public:
	int * index_pool;
	int * lower_limit, * upper_limit;
	int nv;
	BBox bounds;
	float3 bounds_a[2];
	UniformGrid() {
		lower_limit = upper_limit = 0;
		index_pool = 0;
		nv = 0;
		voxelsPerUnitDist = 0;
	};
	__host__ void initialize(int num_triangles);
//	__host__ __device__ void buildGrid(Triangle * p);
	__device__ bool intersect(Triangle * triangles, Ray& ray, float in_coeff);
	__host__ __device__ int posToVoxel(const float pos_comp, int axis);
	__host__ __device__ float voxelToPos(int p, int axis);
	__host__ __device__ int offset(float x, float y, float z);
};

__device__ float3 get_light_color(float3 point, float3 normal, LightSource* l, Triangle* t, float3 viewVector);

//Structures
__host__ __device__ float3 get_point(Ray* r, float t);

#endif
