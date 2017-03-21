#include "structures.h"
#include "utilities.h"

__device__ float3 Triangle::get_normal()
{
	return make_float3(0,0,0);
}

__device__ bool operator==(const float3& v1, const float3& v2)
{
	if (v1.x != v2.x) return false;
	if (v1.y != v2.y) return false;
	if (v1.z != v2.z) return false;
	return true;
}

__device__ bool operator!=(const float3& v1, const float3& v2)
{
	return !(v1==v2);   
}

__device__ float3 operator+(const float3& v1, const float3& v2)
{
	return make_float3(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z);
}

__device__ float3 operator-(const float3& v1, const float3& v2)
{
	return make_float3(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z);   
}

__device__ float3 operator/(const float3& v, float scalar)
{
	return make_float3(v.x/scalar, v.y/scalar, v.z/scalar);   
}

__device__ float3 operator*(const float3& v, float scalar)
{
	return make_float3(v.x*scalar, v.y*scalar, v.z*scalar);       
}

__device__ float3 operator*(float scalar, const float3& v)
{
	return make_float3(v.x*scalar, v.y*scalar, v.z*scalar);       
}

__device__ float3 operator*(const float3& v, const float3& v1)
{
	return make_float3(v.x*v1.x, v.y*v1.y, v.z*v1.z);
}


__device__ float squaredlength(const float3& f)
{ return (f.x*f.x + f.y*f.y + f.z*f.z); }

__device__ float length(const float3& f)
{ return sqrt(squaredlength(f)); }

__device__ float3 normalize(const float3& f)
{ return f / length(f);}

__device__ float3 unitVector(const float3& v)
{
	float len  = length(v);
	return v / len;
}

__device__ float3 crossProduct(const float3& v1, const float3& v2)
{
	float3 tmp;
	tmp.x = v1.y * v2.z - v1.z * v2.y;
	tmp.y = v1.z * v2.x - v1.x * v2.z;
	tmp.z = v1.x * v2.y - v1.y * v2.x;
	return tmp; 
}

__device__ float dotProduct(const float3& v1, const float3& v2)
{ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }

__device__ float tripleProduct(const float3& v1,const float3& v2,const float3& v3)
{
	return dotProduct(( crossProduct(v1, v2)), v3);   
}

__device__ bool Triangle::intersect(Ray *r)
{
	float A = determinant(vertexA-vertexB,vertexA-vertexC,r->direction);
	if(abs(A) < EPSILON) return false;
	float beta = determinant(vertexA-r->origin,vertexA-vertexC,r->direction) / A;
	float gamma = determinant(vertexA-vertexB,vertexA-r->origin,r->direction) / A;
	float t = determinant(vertexA-vertexB,vertexA-vertexC,vertexA-r->origin) / A;
	if(!(beta > 0.0 && gamma > 0.0 && beta+gamma < 1.0)) return false;
	if(!r->has_intersected)	{
		r->has_intersected = true;
		r->t = t;
	}
	else r->t = (r->t)>t?t:r->t;
	return true;
}