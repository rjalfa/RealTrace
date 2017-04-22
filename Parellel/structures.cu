#include "structures.h"
#include "utilities.h"
#include "helper_cuda.h"
#include <cstdio>

const float SICK_FLT_MAX = 1e36;
const float SICK_FLT_MIN = -1e36;
__host__ __device__ bool operator==(const float3& v1, const float3& v2)
{
	if (v1.x != v2.x) return false;
	if (v1.y != v2.y) return false;
	if (v1.z != v2.z) return false;
	return true;
}

__host__ __device__ bool operator!=(const float3& v1, const float3& v2)
{
	return !(v1 == v2);
}

__host__ __device__ float3 operator+(const float3& v1, const float3& v2)
{
	return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__host__ __device__ float3 operator+(const float3& v1, const double& v2)
{
	return make_float3(v1.x + v2, v1.y + v2, v1.z + v2);
}

__host__ __device__ float3 operator-(const float3& v1, const float3& v2)
{
	return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__host__ __device__ float3 operator/(const float3& v, float scalar)
{
	return make_float3(v.x / scalar, v.y / scalar, v.z / scalar);
}

__host__ __device__ float3 operator*(const float3& v, float scalar)
{
	return make_float3(v.x * scalar, v.y * scalar, v.z * scalar);
}

//Unary
__host__ __device__ float3 operator-(const float3& v)
{
	return make_float3(-v.x, -v.y, -v.z);
}

__host__ __device__ float3 operator*(float scalar, const float3& v)
{
	return v * scalar;
}

__host__ __device__ float3 operator*(const float3& v, const float3& v1)
{
	return make_float3(v.x * v1.x, v.y * v1.y, v.z * v1.z);
}

__host__ __device__ float squaredlength(const float3& f)
{
	return (f.x * f.x + f.y * f.y + f.z * f.z);
}

__host__ __device__ float length(const float3& f)
{
	return sqrt((double)squaredlength(f));
}

__host__ __device__ float3 normalize(const float3& f)
{
	return f / length(f);
}

__host__ __device__ float3 unitVector(const float3& v)
{
	float len  = length(v);
	return v / len;
}

__host__ __device__ float3 get_point(Ray* r, float t)
{
	return r->origin + t * r->direction;
}

__host__ __device__ float dotProduct(const float3& v1, const float3& v2)
{ return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }

__host__ __device__ float tripleProduct(const float3& v1, const float3& v2, const float3& v3)
{
	return dotProduct(( crossProduct(v1, v2)), v3);
}

__host__ __device__ float distance(const float3& v1, const float3& v2)
{
	return sqrt(0.0 + (v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z));
}

__host__ __device__ float3 reflect(const float3& I, const float3& N)
{
	return I - 2.0f * dotProduct(N, I) * N;
}

__host__ __device__ float3 crossProduct(const float3& v1, const float3& v2)
{
	float3 tmp;
	tmp.x = v1.y * v2.z - v1.z * v2.y;
	tmp.y = v1.z * v2.x - v1.x * v2.z;
	tmp.z = v1.x * v2.y - v1.y * v2.x;
	return tmp;
}

__host__ __device__ float3 Triangle::get_normal()
{
	return crossProduct(vertexA - vertexB, vertexA - vertexC);
}

__host__ __device__ bool Triangle::intersect(Ray *r)
{
	float A = determinant(vertexA - vertexB, vertexA - vertexC, r->direction);
	if (abs((double) A) < EPSILON) return false;
	float beta = determinant(vertexA - r->origin, vertexA - vertexC, r->direction) / A;
	float gamma = determinant(vertexA - vertexB, vertexA - r->origin, r->direction) / A;
	float t = determinant(vertexA - vertexB, vertexA - vertexC, vertexA - r->origin) / A;
	if (!(beta > 0.0 && gamma > 0.0 && beta + gamma < 1.0)) return false;
	if (t < 1e-5) return false;
	if ((!r->has_intersected) || (r->t > t))	{
		r->has_intersected = true;
		r->t = t;
		r->intersected = this;
	}
	return true;
}


__host__ __device__ BBox Triangle::getWorldBound() {
	BBox temp;
	for (int vno = 0; vno < 3; vno++) {
		temp.axis_min[0] = min(temp.axis_min[0], getVertex(vno).x);
		temp.axis_max[0] = max(temp.axis_max[0], getVertex(vno).x);
		temp.axis_min[1] = min(temp.axis_min[1], getVertex(vno).y);
		temp.axis_max[1] = max(temp.axis_max[1], getVertex(vno).y);
		temp.axis_min[2] = min(temp.axis_min[2], getVertex(vno).z);
		temp.axis_max[2] = max(temp.axis_max[2], getVertex(vno).z);
	}
	return temp;
}

__host__ __device__ void Triangle::getWorldBound(float& xmin, float& xmax, float& ymin, float& ymax, float& zmin, float& zmax) {
	xmin = ymin = zmin = SICK_FLT_MAX;
	xmax = ymax = zmax = SICK_FLT_MIN;
	for (int vno = 0; vno < 3; vno++) {
		xmin = min(xmin, getVertex(vno).x);
		xmax = max(xmax, getVertex(vno).x);
		ymin = min(ymin, getVertex(vno).y);
		ymax = max(ymax, getVertex(vno).y);
		zmin = min(zmin, getVertex(vno).z);
		zmax = max(zmax, getVertex(vno).z);
	}
}

__host__ __device__ bool BBox::intersects(Ray& ray) {
	float3 bounds_a[2];
	bounds_a[0] = make_float3(this->axis_min[0], this->axis_min[1], this->axis_min[2]);
	bounds_a[1] = make_float3(this->axis_max[0], this->axis_max[1], this->axis_max[2]);

	bool flag = false;
	float dirx = ray.direction.x, diry = ray.direction.y, dirz = ray.direction.z;
	{
		float tmin, tmax, tymin, tymax, tzmin, tzmax;
		bool signx = (dirx < 0), signy = (diry < 0), signz = (dirz < 0);
		tmin = (bounds_a[signx].x - ray.origin.x) / dirx;
		tmax = (bounds_a[1 - signx].x - ray.origin.x) / dirx;
		tymin = (bounds_a[signy].y - ray.origin.y) / diry;
		tymax = (bounds_a[1 - signy].y - ray.origin.y) / diry;
		if (tmin > tymax || tymin > tmax) flag = false;
		if (tymin > tmin) tmin = tymin;
		if (tymax < tmax) tmax = tymax;

		tzmin = (bounds_a[signz].z - ray.origin.z) / dirz;
		tzmax = (bounds_a[1 - signz].z - ray.origin.z) / dirz;

		if ((tmin > tzmax) || (tzmin > tmax)) flag = false;

		flag = true;
	}

	return flag;
}

__host__ __device__ bool BVHTree::checkIntersect(Ray& ray, int idx) {
	if(idx == -1) return false;
	return bbox[idx].intersects(ray);
}

__host__ __device__ void BVHTree::intersect(Triangle * triangles, Ray& ray, int idx) {
	if(idx == -1) return;
	if(checkIntersect(ray, idx)) {
		if(isLeaf[idx]) {
			triangles[primitive_idx[idx]].intersect(&ray);
			return;
		}
		intersect(triangles, ray, left[idx]);
		intersect(triangles, ray, right[idx]);
	}
	return;
}

__host__ __device__ void BVHTree::intersect(Triangle * triangles, Ray& ray) {
	// Allocate traversal stack from thread-local memory,
	// and push NULL to indicate that there are no postponed nodes.
	int stack[64];
	int * stackPtr = stack;
	*stackPtr++ = -1; // push

	// Traverse nodes starting from the root.
	int node = 0;
	do {
		// Check each child node for overlap.
		int  childL = left[node];
		int  childR = right[node];
		bool overlapL = checkIntersect(ray, childL);
		bool overlapR = checkIntersect(ray, childR);

		// Query overlaps a leaf node => report collision.
		bool isLeafL = (childL != -1) && isLeaf[childL];
		bool isLeafR = (childR != -1) && isLeaf[childR];
		if (overlapL && isLeafL)
			triangles[primitive_idx[childL]].intersect(&ray);

		if (overlapR && isLeafR)
			triangles[primitive_idx[childR]].intersect(&ray);

		// Query overlaps an internal node => traverse.
		bool traverseL = (overlapL && !isLeafL);
		bool traverseR = (overlapR && !isLeafR);

		if (!traverseL && !traverseR) {
			node = *--stackPtr; // pop
		} else {
			node = (traverseL) ? childL : childR;
			if (traverseL && traverseR)
				*stackPtr++ = childR; // push
		}
	} while (node != -1);
}

__host__ __device__ float3 get_light_color(float3 point, float3 normal, LightSource* l, Triangle* t, float3 viewVector)
{
	float3 vLightPosition = l->position;
	float3 n = normalize(normal);
	float3 r = normalize(reflect(-normalize(vLightPosition - point), n));
	float dist = ::distance(point, vLightPosition);
	//float fatt = 1.0 / (1.0 + 0.05*dist);
	float diffuse = max(dotProduct(n, normalize(vLightPosition)), 0.0f);
	float specular = max(pow(dotProduct(normalize(viewVector), r), 128), 0.0);
	return 0.8 * diffuse * (l->color) * (t->color) + 0.1 * specular * (l->color);
}
