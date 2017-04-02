#include "structures.h"
#include "utilities.h"

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
	return !(v1==v2);   
}

__host__ __device__ float3 operator+(const float3& v1, const float3& v2)
{
	return make_float3(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z);
}

__host__ __device__ float3 operator+(const float3& v1, const double& v2)
{
	return make_float3(v1.x+v2, v1.y + v2, v1.z + v2);
}

__host__ __device__ float3 operator-(const float3& v1, const float3& v2)
{
	return make_float3(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z);   
}

__host__ __device__ float3 operator/(const float3& v, float scalar)
{
	return make_float3(v.x/scalar, v.y/scalar, v.z/scalar);   
}

__host__ __device__ float3 operator*(const float3& v, float scalar)
{
	return make_float3(v.x*scalar, v.y*scalar, v.z*scalar);       
}

//Unary
__host__ __device__ float3 operator-(const float3& v)
{
	return make_float3(-v.x, -v.y, -v.z);
}

__host__ __device__ float3 operator*(float scalar, const float3& v)
{
	return v*scalar;       
}

__host__ __device__ float3 operator*(const float3& v, const float3& v1)
{
	return make_float3(v.x*v1.x, v.y*v1.y, v.z*v1.z);
}

__host__ __device__ float squaredlength(const float3& f)
{ 
	return (f.x*f.x + f.y*f.y + f.z*f.z);
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
	return r->origin + t*r->direction;
}

__host__ __device__ float dotProduct(const float3& v1, const float3& v2)
{ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }

__host__ __device__ float tripleProduct(const float3& v1,const float3& v2,const float3& v3)
{
	return dotProduct(( crossProduct(v1, v2)), v3);   
}

__host__ __device__ float distance(const float3& v1, const float3& v2)
{
	return sqrt(0.0L + (v1.x-v2.x)*(v1.x-v2.x) + (v1.y-v2.y)*(v1.y-v2.y) + (v1.z-v2.z)*(v1.z-v2.z));
}

__host__ __device__ float3 reflect(const float3& I, const float3& N)
{
	return I - 2.0f*dotProduct(N,I)*N;
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
	return crossProduct(vertexA-vertexB,vertexA-vertexC);
}

__host__ __device__ bool Triangle::intersect(Ray *r)
{
	float A = determinant(vertexA-vertexB,vertexA-vertexC,r->direction);
	if(abs((double) A) < EPSILON) return false;
	float beta = determinant(vertexA-r->origin,vertexA-vertexC,r->direction) / A;
	float gamma = determinant(vertexA-vertexB,vertexA-r->origin,r->direction) / A;
	float t = determinant(vertexA-vertexB,vertexA-vertexC,vertexA-r->origin) / A;
	if(!(beta > 0.0 && gamma > 0.0 && beta+gamma < 1.0)) return false;
	if(t < 1e-5) return false;
	if(!r->has_intersected)	{
		r->has_intersected = true;
		r->t = t;
		r->intersected = this;
	}
	else if(r->t > t)
	{
		r->t = t;
		r->intersected = this;
	}
	return true;
}


__host__ __device__ void Voxel::addPrimitive(int req_idx, int i) {
	// cout << p << endl;
	// idx.push_back(i);
	// primitives.push_back(p);
	primitives[req_idx] = i;
}

__host__ __device__ bool Voxel::intersect(Triangle * triangles, Ray& ray) {
	bool hitSomething = false;
	for(int i = 0; i < max_size; i++) {
		if(triangles[primitives[i]].intersect(&ray)) {
//			ray.setIdx(idx[i]);
			hitSomething = true;
		}
	}
	return hitSomething;
}

__host__ __device__ float UniformGrid::findVoxelsPerUnitDist(float delta[], int num) {
	float maxAxis = max(delta[0], max(delta[1], delta[2]));
	float invMaxWidth = 1.0 / maxAxis;
	float cubeRoot = 3.0 * pow(num, 1.0 / 3.0);
	float voxelsPerUnitDist = cubeRoot * invMaxWidth;
	return voxelsPerUnitDist;
}

__host__ __device__ int UniformGrid::posToVoxel(const float3& pos, int axis) {
	float val;
	if(axis == 0) val = pos.x;
	else if(axis == 2) val = pos.y;
	else val = pos.z;
	int v = ((val - bounds.axis_min[axis]) * invWidth[axis]);
	return min(max(v, 0), nVoxels[axis] - 1);
}

__host__ __device__ float UniformGrid::voxelToPos(int p, int axis) {
	return bounds.axis_min[axis] + p * width[axis];
}

__host__ __device__ inline int UniformGrid::offset(float x, float y, float z) {
	return z * nVoxels[0] * nVoxels[1] + y * nVoxels[0] + x;
}

__host__ void UniformGrid::initialize(int num_triangles) {
	// assuming triangle objects do not require more refinements
	
	// for(int i = 0; i < p.size(); i++) {
	// 	for(int j = 0; j < 3; j++) {
	// 		BBox temp = p[i]->getWorldBound();
	// 		for(int axis = 0; axis < 3; axis++) {
	// 			bounds.axis_min[axis] = min(bounds.axis_min[axis], temp.axis_min[axis]);
	// 			bounds.axis_max[axis] = max(bounds.axis_max[axis], temp.axis_max[axis]);
	// 		}
	// 	}
	// }
		
	for(int axis = 0; axis < 3; axis++)
		delta[axis] = (bounds.axis_max[axis] - bounds.axis_min[axis]);

	// find voxelsPerUnitDist
	voxelsPerUnitDist = findVoxelsPerUnitDist(delta, num_triangles);

	for(int axis = 0; axis < 3; axis++) {
		nVoxels[axis] = ceil(0.0L + delta[axis] * voxelsPerUnitDist);
		nVoxels[axis] = max(1, nVoxels[axis]);
		// 64 is magically determined number, lol
		nVoxels[axis] = min(nVoxels[axis], 64);
	}

	nv = 1;
	for(int axis = 0; axis < 3; axis++) {
		width[axis] = delta[axis] / nVoxels[axis];
		invWidth[axis] = (width[axis] == 0.0) ? 0.0 : 1.0 / width[axis];
		nv *= nVoxels[axis];
	}

	cudaMalloc(&voxels, sizeof(Voxel) * nv);
	cudaMalloc(&voxel_sizes, sizeof(int) * nv);
}


//__host__ __device__ void UniformGrid::buildGrid(Triangle * p) {
//
//	for(int i = 0; i < num_triangles; i++) {
//		BBox pb = p[i].getWorldBound();
//		int vmin[3], vmax[3];
//		for(int axis = 0; axis < 3; axis++) {
//			vmin[axis] = posToVoxel(make_float3(pb.axis_min[0], pb.axis_min[1], pb.axis_min[2]), axis);
//			vmax[axis] = posToVoxel(make_float3(pb.axis_max[0], pb.axis_max[1], pb.axis_max[2]), axis);
//		}
//
//		for(int z = vmin[2]; z <= vmax[2]; z++) {
//			for(int y = vmin[1]; y <= vmax[1]; y++) {
//				for(int x = vmin[0]; x <= vmax[0]; x++) {
//					int o = offset(x, y, z);
//					int req_idx = atomicAdd(voxels[o].curr_size, 1);
//					voxels[req_idx].addPrimitive(p[i], i);
//				}
//			}
//		}
//	}
//}

__host__ __device__ BBox Triangle::getWorldBound() {
	BBox temp;
	for(int vno = 0; vno < 3; vno++) {
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
	for(int vno = 0; vno < 3; vno++) {
		xmin = min(xmin, getVertex(vno).x);
		xmax = max(xmax, getVertex(vno).x);
		ymin = min(ymin, getVertex(vno).y);
		ymax = max(ymax, getVertex(vno).y);
		zmin = min(zmin, getVertex(vno).z);
		zmax = max(zmax, getVertex(vno).z);
	}
}

template <typename T>
__host__ __device__  inline void do_swap (T& a, T& b)  {
   T x = a;
   a = b;
   b = x;
}

__host__ __device__ bool UniformGrid::intersect(Triangle * triangles, Ray& ray) {
	// check ray against overall grid bounds
	float rayT;
	bool flag = true;
	{
		float tmin = (bounds.axis_min[0] - ray.origin.x) / ray.direction.x;
		float tmax = (bounds.axis_max[0] - ray.origin.x) / ray.direction.x;
		if(tmin > tmax) do_swap(tmin, tmax);
		 
		float tymin = (bounds.axis_min[1] - ray.origin.y) / ray.direction.y;
		float tymax = (bounds.axis_max[1] - ray.origin.y) / ray.direction.y;

		if (tymin > tymax) do_swap(tymin, tymax);

		if ((tmin > tymax) || (tymin > tmax)) 
		    flag = false;

		if (tymin > tmin) 
		    tmin = tymin; 

		if (tymax < tmax) 
		    tmax = tymax; 

		float tzmin = (bounds.axis_min[2] - ray.origin.z) / ray.direction.z;
		float tzmax = (bounds.axis_max[2] - ray.origin.z) / ray.direction.z;

		if (tzmin > tzmax) do_swap(tzmin, tzmax);

		if ((tmin > tzmax) || (tzmin > tmax)) 
		    flag = false;

		if (tzmin > tmin) 
		    tmin = tzmin; 

		if (tzmax < tmax) 
		    tmax = tzmax; 

		rayT = tmin;
		if(flag) {
			ray.strictSetParameter(rayT);
		}
	}

	if(!flag) return false;

	float3 gridIntersect = get_point(&ray, ray.t);

	int pos[3], step[3], out[3];
	float nextCrossingT[3], deltaT[3];
	// set up 3D DDA for ray
	for(int axis = 0; axis < 3; axis++) {
		// compute current voxel for axis
		pos[axis] = posToVoxel(gridIntersect, axis);
		if(ray.direction.x >= 0) {
			// handle ray with positive direction for voxel stepping
			nextCrossingT[axis] = rayT + (voxelToPos(pos[axis] + 1, axis) - gridIntersect.x) / ray.direction.x;
			deltaT[axis] = width[axis] / ray.direction.x;
			step[axis] = 1;
			out[axis] = nVoxels[axis];
		} else {
			// handle ray with negative direction for voxel stepping
			nextCrossingT[axis] = rayT + (voxelToPos(pos[axis], axis) - gridIntersect.x) / ray.direction.x;
			deltaT[axis] = -width[axis] / ray.direction.x;
			step[axis] = -1;
			out[axis] = -1;
		}
		if(ray.direction.y >= 0) {
			// handle ray with positive direction for voxel stepping
			nextCrossingT[axis] = rayT + (voxelToPos(pos[axis] + 1, axis) - gridIntersect.y) / ray.direction.y;
			deltaT[axis] = width[axis] / ray.direction.y;
			step[axis] = 1;
			out[axis] = nVoxels[axis];
		} else {
			// handle ray with negative direction for voxel stepping
			nextCrossingT[axis] = rayT + (voxelToPos(pos[axis], axis) - gridIntersect.y) / ray.direction.y;
			deltaT[axis] = -width[axis] / ray.direction.y;
			step[axis] = -1;
			out[axis] = -1;
		}
		if(ray.direction.z >= 0) {
			// handle ray with positive direction for voxel stepping
			nextCrossingT[axis] = rayT + (voxelToPos(pos[axis] + 1, axis) - gridIntersect.z) / ray.direction.z;
			deltaT[axis] = width[axis] / ray.direction.z;
			step[axis] = 1;
			out[axis] = nVoxels[axis];
		} else {
			// handle ray with negative direction for voxel stepping
			nextCrossingT[axis] = rayT + (voxelToPos(pos[axis], axis) - gridIntersect.z) / ray.direction.z;
			deltaT[axis] = -width[axis] / ray.direction.z;
			step[axis] = -1;
			out[axis] = -1;
		}
	}
	// walk ray through voxel grid
	bool hitSomething = false;
	ray.strictSetParameter(SICK_FLT_MAX);
	for( ; ; ) {
		// check for intersection in current voxel and advance to next
		// Voxel * voxel = voxels[offset(pos[0], pos[1], pos[2])];
		Voxel& voxel = voxels[offset(pos[0], pos[1], pos[2])];
		if(voxel.max_size != 0)
			hitSomething |= voxel.intersect(triangles, ray);
		// advance to next voxel
		// find stepAxis for stepping to next voxel
		int bits =  ((nextCrossingT[0] < nextCrossingT[1]) << 2) +
					((nextCrossingT[0] < nextCrossingT[2]) << 1) +
					((nextCrossingT[1] < nextCrossingT[2]));
		const int cmpToAxis[8] = {2, 1, 2, 1, 2, 2, 0, 0};
		int stepAxis = cmpToAxis[bits];

		pos[stepAxis] += step[stepAxis];
		if(pos[stepAxis] == out[stepAxis])
			break;
		if(hitSomething) break;
		nextCrossingT[stepAxis] += deltaT[stepAxis];
	}
	return hitSomething;
}

__host__ __device__ float3 get_light_color(float3 point, float3 normal, LightSource* l, Triangle* t, float3 viewVector)
{
	float3 vLightPosition = l->position;
	float3 n = normalize(normal);
	float3 r = normalize(reflect(-normalize(vLightPosition-point),n));
	float dist = ::distance(point,vLightPosition);
	//float fatt = 1.0 / (1.0 + 0.05*dist);
	float diffuse = max(dotProduct(n,normalize(vLightPosition)),0.0f);
	float specular = max(pow(dotProduct(normalize(viewVector),r),128),0.0);
	return 0.8*diffuse*(l->color)*(t->color) + 0.1*specular*(l->color);
}
