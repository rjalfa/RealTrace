#ifndef __STRUCTURES_H
#define __STRUCTURES_H
#include <thrust/device_vector.h>
#include <vector>

using namespace std;

class Triangle;

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
		__host__ __device__ float3 get_normal();
		__host__ __device__ bool intersect(Ray *r);
		__host__ __device__ BBox getWorldBound();
		__host__ __device__ void getWorldBound(float& xmin, float& xmax, float& ymin, float& ymax, float& zmin, float& zmax);
		__host__ __device__ float3 getVertex(int vno) {
			if(vno == 0) return vertexA;
			else if(vno == 1) return vertexB;
			else return vertexC;
		}
};

class Voxel {
public:
	int curr_size, max_size;
	int * primitives;
	__host__ __device__ void addPrimitive(int req_idx, int i);
	__host__ __device__ bool intersect(Triangle * triangles, Ray& ray);
	Voxel() {
		curr_size = 0;
		primitives = 0;
		max_size = 0;
	}
};

class UniformGrid {
	float delta[3];
	int nVoxels[3];
	float voxelsPerUnitDist;
	float width[3], invWidth[3];
	// thrust::device_vector < Voxel > voxels;

	__host__ __device__ float findVoxelsPerUnitDist(float delta[], int num);

public:
	int * voxel_sizes;
	Voxel * voxels;
	int nv;
	BBox bounds;
	UniformGrid() {
		voxel_sizes = 0;
		voxels = 0;
		nv = 0;
		voxelsPerUnitDist = 0;
	};
	__host__ void initialize(int num_triangles);
//	__host__ __device__ void buildGrid(Triangle * p);
	__host__ __device__ bool intersect(Triangle * triangles, Ray& ray);
	__host__ __device__ int posToVoxel(const float3& pos, int axis);
	__host__ __device__ float voxelToPos(int p, int axis);
	__host__ __device__ int offset(float x, float y, float z);
};

__host__ __device__ float3 get_light_color(float3 point, float3 normal, LightSource* l, Triangle* t, float3 viewVector);

//Structures
__host__ __device__ bool operator==(const float3& v1, const float3& v2);
__host__ __device__ bool operator!=(const float3& v1, const float3& v2);
__host__ __device__ float3 operator+(const float3& v1, const float3& v2);
__host__ __device__ float3 operator+(const float3& v1, const double& v2);
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
