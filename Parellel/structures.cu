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

//Unary
__device__ float3 operator-(const float3& v)
{
	return make_float3(-v.x, -v.y, -v.z);
}

__device__ float3 operator*(float scalar, const float3& v)
{
	return v*scalar;       
}

__device__ float3 operator*(const float3& v, const float3& v1)
{
	return make_float3(v.x*v1.x, v.y*v1.y, v.z*v1.z);
}

__device__ float squaredlength(const float3& f)
{ 
	return (f.x*f.x + f.y*f.y + f.z*f.z);
}

__device__ float length(const float3& f)
{
	return sqrt(squaredlength(f));
}

__device__ float3 normalize(const float3& f)
{
	return f / length(f);
}

__device__ float3 unitVector(const float3& v)
{
	float len  = length(v);
	return v / len;
}

__device__ float3 get_point(Ray* r, float t)
{
	return r->origin + t*r->direction;
}

__device__ float dotProduct(const float3& v1, const float3& v2)
{ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }

__device__ float tripleProduct(const float3& v1,const float3& v2,const float3& v3)
{
	return dotProduct(( crossProduct(v1, v2)), v3);   
}

__device__ float distance(const float3& v1, const float3& v2)
{
	return sqrt((v1.x-v2.x)*(v1.x-v2.x) + (v1.y-v2.y)*(v1.y-v2.y) + (v1.z-v2.z)*(v1.z-v2.z));
}

__device__ float3 reflect(const float3& I, const float3& N)
{
	return I - 2.0f*dotProduct(N,I)*N;
}

__device__ float3 crossProduct(const float3& v1, const float3& v2)
{
	float3 tmp;
	tmp.x = v1.y * v2.z - v1.z * v2.y;
	tmp.y = v1.z * v2.x - v1.x * v2.z;
	tmp.z = v1.x * v2.y - v1.y * v2.x;
	return tmp; 
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


__device__ float3 get_light_color(float3 point, float3 normal, LightSource* l, Triangle* t, float3 viewVector)
{
	float3 vLightPosition = l->position;
	float3 n = normalize(normal);
	float3 r = normalize(reflect(-normalize(vLightPosition-point),n));
	float dist = distance(point,vLightPosition);
	//float fatt = 1.0 / (1.0 + 0.05*dist);
	float diffuse = max(dotProduct(n,normalize(vLightPosition)),0.0f);
	float specular = max(pow(dotProduct(normalize(viewVector),r),128),0.0);
	return 0.8*diffuse*(l->color)*(t->color) + 0.1*specular*(l->color);
}