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
//	xmin = ymin = zmin = std::numeric_limits < float >::max();
//	xmax = ymax = zmax = std::numeric_limits < float >::min();
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

__host__ __device__ bool UniformGrid::intersect(Triangle * triangles, Ray& ray) {
	// check ray against overall grid bounds

	float rayT;
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

		if (tzmin > tmin) tmin = tzmin;

		if (tzmax < tmax) tmax = tzmax;

		rayT = tmin;
		ray.strictSetParameter(rayT);

	}

	float3 gridIntersectf3 = get_point(&ray, ray.t);
	float gridIntersect[3];
	gridIntersect[0] = gridIntersectf3.x;
	gridIntersect[1] = gridIntersectf3.y;
	gridIntersect[2] = gridIntersectf3.z;
	float direction[3];
	direction[0] = ray.direction.x;
	direction[1] = ray.direction.y;
	direction[2] = ray.direction.z;
	int pos[3], step[3], out[3];
	float nextCrossingT[3], deltaT[3];
	// set up 3D DDA for ray
	for (int axis = 0; axis < 3; axis++) {
		// compute current voxel for axis
		pos[axis] = posToVoxel(gridIntersect[axis], axis);
		if (direction[axis] >= 0) {
			// handle ray with positive direction for voxel stepping
			nextCrossingT[axis] = rayT + (voxelToPos(pos[axis] + 1, axis) - gridIntersect[axis]) / direction[axis];
			deltaT[axis] = width[axis] / direction[axis];
			step[axis] = 1;
			out[axis] = nVoxels[axis];
		} else {
			// handle ray with negative direction for voxel stepping
			nextCrossingT[axis] = rayT + (voxelToPos(pos[axis], axis) - gridIntersect[axis]) / direction[axis];
			deltaT[axis] = -width[axis] / direction[axis];
			step[axis] = -1;
			out[axis] = -1;
		}
		// cerr << pos[axis] << " " << step[axis] << " " << out[axis] << endl;
	}
	// walk ray through voxel grid
	bool hitSomething = false;
	ray.strictSetParameter(SICK_FLT_MAX);
	for ( ; ; ) {
		// check for intersection in current voxel and advance to next
		// Voxel * voxel = voxels[offset(pos[0], pos[1], pos[2])];
		int voxel = offset(pos[0], pos[1], pos[2]);

		hitSomething |= Voxel::intersect(this, triangles, ray, voxel);
		// advance to next voxel
		// find stepAxis for stepping to next voxel
		int bits =  ((nextCrossingT[0] < nextCrossingT[1]) << 2) +
					((nextCrossingT[0] < nextCrossingT[2]) << 1) +
					((nextCrossingT[1] < nextCrossingT[2]));
		const int cmpToAxis[8] = {2, 1, 2, 1, 2, 2, 0, 0};
		int stepAxis = cmpToAxis[bits];

		pos[stepAxis] += step[stepAxis];
		if (pos[stepAxis] == out[stepAxis]) break;
		if (hitSomething) break;
		nextCrossingT[stepAxis] += deltaT[stepAxis];
	}
	return hitSomething;
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
