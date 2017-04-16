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
__host__ __device__ float tripleProduct(const float3& v1, const float3& v2, const float3& v3);

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
		t = -1;
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
		axis_min[0] = axis_min[1] = axis_min[2] = 1e36;
		axis_max[0] = axis_max[1] = axis_max[2] = -1e36;
	}
	__host__ __device__
	void doUnion(const BBox& temp) {
		for(int i = 0; i < 3; i++) {
			axis_max[i] = max(axis_max[i], temp.axis_max[i]);
			axis_min[i] = min(axis_min[i], temp.axis_min[i]);
		}
	}
};

class Triangle
{
public:
	float3 vertexA;
	float3 vertexB;
	float3 vertexC;
	float3 color;
	int type_of_material;
	__host__ __device__ float3 get_normal();
	__host__ __device__ bool intersect(Ray *r);
	__host__ __device__ BBox getWorldBound();
	__host__ __device__ void getWorldBound(float& xmin, float& xmax, float& ymin, float& ymax, float& zmin, float& zmax);
	__host__ __device__ float3 getVertex(int vno) {
		if (vno == 0) return vertexA;
		else if (vno == 1) return vertexB;
		else return vertexC;
	}
};

__host__ __device__ bool BBoxIntersects(BBox& box, Ray& ray) {
	float3 bounds_a[2];
	bounds_a[0] = make_float3(box.axis_min[0], box.axis_min[1], box.axis_min[2]);
	bounds_a[1] = make_float3(box.axis_max[0], box.axis_max[1], box.axis_max[2]);
	float dirx = ray.direction.x, diry = ray.direction.y, dirz = ray.direction.z;
	{
		float tmin, tmax, tymin, tymax, tzmin, tzmax;
		bool signx = (dirx < 0), signy = (diry < 0), signz = (dirz < 0);
		tmin = (bounds_a[signx].x - ray.origin.x) / dirx;
		tmax = (bounds_a[1 - signx].x - ray.origin.x) / dirx;
		tymin = (bounds_a[signy].y - ray.origin.y) / diry;
		tymax = (bounds_a[1 - signy].y - ray.origin.y) / diry;
		if (tmin > tymax || tymin > tmax) return false;
		if (tymin > tmin) tmin = tymin;
		if (tymax < tmax) tmax = tymax;

		tzmin = (bounds_a[signz].z - ray.origin.z) / dirz;
		tzmax = (bounds_a[1 - signz].z - ray.origin.z) / dirz;

		if ((tmin > tzmax) || (tzmin > tmax)) return false;
	}
	return true;
}

class BVHNode;

class BVHNode {
public:
    BVHNode * left, * right;
    int first, last;
    BBox bbox;

    __host__ __device__ BVHNode() {
        left = right = NULL;
        first = last = -1;
    }

    __host__ __device__ BVHNode(BVHNode * left, BVHNode * right, int first, int last) {
        this->first = first;
        this->last = last;
        this->left = left;
        this->right = right;
    }

    __host__ __device__ void calcBBox() {
        if(left != NULL && right != NULL)
            left->bbox.doUnion(right->bbox);
    }

    __host__ __device__ void intersect(Triangle * triangles, Ray& ray) {
        // check if ray intersects with its bbox, otherwise exit
        // for(int i = first; i <= last; i++)
        //     triangles[i].intersect(ray);
    	if(::BBoxIntersects(bbox, ray)) {
    		if(first == last) {
    			triangles[first].intersect(&ray);
    		} else {
    			left->intersect(triangles, ray);
    			right->intersect(triangles, ray);
    		}
    	}
    }
};

__host__ __device__ float3 get_light_color(float3 point, float3 normal, LightSource* l, Triangle* t, float3 viewVector);

//Structures
__host__ __device__ float3 get_point(Ray* r, float t);

#endif
