#include "kernel.h"
#include "structures.h"
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include "camera.h"
#include "helper_cuda.h"
#include "cuda_profiler_api.h"
#include "utilities.h"
#include <queue>

#define TX 32
#define TY 32
#define AMBIENT_COLOR make_float3(0.235294, 0.67451, 0.843137)
#define KR 0.001
#define KT 0.1
#define EULER_CONSTANT 2.718
#define eta 4.0
#define KA 0.4

__device__ unsigned char clip(float x) { return x > 255 ? 255 : (x < 0 ? 0 : x); }

BVHTree * d_tree;
float3* colors = 0;

Ray* d_rays[7];
float* d_coeffs[7];
float** d_d_coeffs = NULL;
cudaEvent_t event;
cudaStream_t streamA1, streamA2, streamA3, streamA4;


// kernel function to compute decay and shading
__device__ void get_color_from_float3(float3 color, uchar4* cell)
{
	cell->x = clip(color.x * 255);
	cell->y = clip(color.y * 255);
	cell->z = clip(color.z * 255);
	cell->w = 255;
}

__device__ bool refract(const float3& I, const float3& N, const float e, float3& T)
{
	float k = 1.0 - e * e * (1.0 - dotProduct(N, I) * dotProduct(N, I));
	if (k < 0) return false;
	T = e * I - (e * dotProduct(N, I) + sqrt(k)) * N;
	return true;
}

__device__ void fresnel(const float3& I, const float3& N, const float& ior, float &kr)
{
	float cosi = clamp(-1, 1, dotProduct(I, N));
	float etai = 1, etat = ior;
	if (cosi > 0) {
		float t = etai;
		etai = etat;
		etat = t;
	}
	// Compute sini using Snell's law
	float sint = etai / etat * sqrtf(max(0.f, 1 - cosi * cosi));
	// Total internal reflection
	if (sint >= 1) {
		kr = 1;
	}
	else {
		float cost = sqrtf(max(0.f, 1 - sint * sint));
		cosi = fabsf(cosi);
		float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
		float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
		kr = (Rs * Rs + Rp * Rp) / 2;
	}
}

//Shared Memory Loop Intersect

__device__ void intersect(Triangle* triangles, int num_triangles, Ray* r, BVHTree * root)
{
//  __shared__ Triangle localObjects[32];
//  int triangles_to_scan = num_triangles;
//  while(triangles_to_scan > 0)
//  {
//    int x = min(triangles_to_scan,32);
//    if(threadIdx.x == 0 && threadIdx.y < x) localObjects[threadIdx.y] = triangles[threadIdx.y];
//    __syncthreads();
//
//    for(int i = 0; i < x; i ++) localObjects[i].intersect(r);
//    triangles += 32;
//    triangles_to_scan -= 32;
//    __syncthreads();
//  }
	root->intersect(triangles, *r, 0);
}

////////////////////////////////////////////////////////////////////////////
// Ray generation kernel
// Parameters:
// camera = Camera object
// w = width
// h = height
// out_rays = Output rays
// d_out = Output image to be resetted
////////////////////////////////////////////////////////////////////////////
__global__ void createRaysAndResetImage(Camera* camera, int w, int h, Ray* out_rays, uchar4* d_out, float* d_coeffs[7], float3* out_color)
{
	if (!camera || !out_rays || !d_out) return;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	float3 pos = camera->get_position();
	float3 dir = camera->get_ray_direction(i, j);
	int index = i + j * w; // 1D indexing
	out_rays[index] = Ray(pos, dir);
	d_out[index] = make_uchar4(0, 0, 0, 0);

	out_color[index] = make_float3(0, 0, 0);
	for (int i = 1; i < 7; i ++)
	{
		if (d_coeffs[i] != NULL) d_coeffs[i][index] = 0.0f;
	}
}

////////////////////////////////////////////////////////////////////////////
// Recursive Ray-tracing Kernel
// Parameters:
// out_color = Global Color Array that stores output from all kernels
// in_coeffs = The coeffs for the current kernel rays. If NULL, assumed all 1's
// w = width
// h = height
// rays = The rays to trace for this kernel
// out_rays_reflect = The rays that emerge from reflection from this kernel, If NULL, assumed end of recursion
// out_rays_refract = The rays that emerge from reflection from this kernel, If NULL, assumed end of recursion
// out_coeffs_reflect = The coeffs for the reflected rays
// out_coeffs_refract = The coeffs for the refracted rays
// triangles = Triangle objects
// num_triangles = Number of triangles in above
// l = LightSource object
// ug = UniformGrid object
////////////////////////////////////////////////////////////////////////////
__global__ void raytrace(float3 *out_color, float* in_coeffs, int w, int h, Ray* rays, Ray* out_rays_reflect, float* out_coeffs_reflect, Ray* out_rays_refract, float* out_coeffs_refract, Triangle* triangles, int num_triangles, LightSource* l, BVHTree * root)
{
	if (out_color == NULL || rays == NULL) return;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int index = i + j * w;
	//Switches
	bool in_coeff = (in_coeffs != NULL) ? in_coeffs[index] : 1.00;

	if (in_coeff < EPSILON || rays[index].direction == make_float3(0, 0, 0)) return;

	bool can_refract = (out_rays_refract != NULL && out_coeffs_refract != NULL);
	bool can_reflect = (out_rays_reflect != NULL && out_coeffs_reflect != NULL);
	bool will_refract = false;
	bool will_reflect = false;
	//Get owned ray
	Ray ray = rays[index];
	intersect(triangles, num_triangles, &ray, root);
	//bool reflect_over_refract = false;
	//Do one time intersection
	float3 finalColor = make_float3(0, 0, 0);
	if (!ray.has_intersected) finalColor = AMBIENT_COLOR;
	else
	{
		float3 I = normalize(ray.direction);
		float3 N = normalize(ray.intersected->get_normal());
		will_reflect = (ray.intersected->type_of_material == REFLECTIVE);
		will_refract = (ray.intersected->type_of_material == REFRACTIVE);
		finalColor = get_light_color(get_point(&ray, ray.t), N, l, ray.intersected, I);
		finalColor = finalColor + (ray.intersected)->color * KA;
		if ((!can_reflect && !can_refract) || (!will_reflect && !will_refract)) {  }
		//Reflect
		else if (can_reflect && will_reflect)
		{
			float3 R = reflect(I, N);
			Ray reflectedRay(ray.getPosition() + 1e-4 * R, R);
			out_rays_reflect[index] = reflectedRay;
			out_coeffs_reflect[index] = in_coeff * KR;
			finalColor = finalColor * (1 - KR);
		}
		else if (can_refract && will_refract)
		{
			/*
			float c = 0;//,k = 0;
			float3 R = reflect(I,N);
			float3 T;
			if(dotProduct(I,N) < 0)
			{
			  refract(I,N,eta,T);
			  c = -dotProduct(I,N);
			}
			else
			{
			  //k = 1;
			  //k = make_float3(pow(EULER_CONSTANT,-1.0*0.27*t),pow(EULER_CONSTANT,-1.0*0.45*t),pow(EULER_CONSTANT,-1.0*0.55*t));
			  if(refract(I,-1.0*N,1/eta,T)) c = dotProduct(T,N);
			  else {
				Ray reflectedRay(ray.getPosition()+ 1e-4 * R,R);
				out_rays_reflect[index] = reflectedRay;
				out_coeffs_reflect[index] = in_coeff * KR;
				finalColor = finalColor * (1-KR);
				//return k*shade_ray(temp);
				reflect_over_refract = true;
			  }
			}
			if(!reflect_over_refract)
			{
			  float _R0 = ((eta-1)*(eta-1))/((eta+1)*(eta+1));
			  float _R = _R0 + (1-_R0)*pow(1-c,5);
			  Ray temp1 = Ray(ray.getPosition()+ 1e-4 * R,R);
			  Ray temp2 = Ray(ray.getPosition()+ 1e-4 * T,T);
			  out_rays_reflect[index] = temp1;
			  out_coeffs_reflect[index] = in_coeff * _R;
			  out_rays_refract[index] = temp2;
			  out_coeffs_refract[index] = in_coeff * (1-_R);
			  in_coeff = 0;
			}
			*/
			float3 refractionColor = make_float3(0, 0, 0);
			// compute fresnel
			float kr;
			float3 hitPoint = ray.getPosition();
			fresnel(I, N, eta, kr);
			bool outside = (dotProduct(I, N) < 0);
			float3 bias = N * 1e-4f;
			// compute refraction if it is not a case of total internal reflection
			if (kr < 1) {
				float3 refractionDirection;
				refract(I, N, eta, refractionDirection);
				refractionDirection = normalize(refractionDirection);
				float3 refractionRayOrig = outside ? hitPoint - bias : hitPoint + bias;
				Ray refractedRay(refractionRayOrig, refractionDirection);
				out_rays_refract[index] = refractedRay;
				out_coeffs_refract[index] = in_coeff * (1 - kr);
				//refractionColor = castRay(refractionRayOrig, refractionDirection, objects, lights, options, depth + 1);
			}
			float3 reflectionDirection = normalize(reflect(I, N));
			float3 reflectionRayOrig = outside ? hitPoint + bias : hitPoint - bias;
			//float3 reflectionColor = castRay(reflectionRayOrig, reflectionDirection, objects, lights, options, depth + 1);

			out_rays_refract[index] = Ray(reflectionRayOrig, reflectionDirection);
			out_coeffs_refract[index] = in_coeff * kr;

			// mix the two
			//finalColor += reflectionColor * kr + refractionColor * (1 - kr);
			in_coeff = 0.0;
		}

	}
	finalColor = finalColor * in_coeff;
	// out_color[index] = finalColor;
	atomicAdd(&out_color[index].x, finalColor.x);
	atomicAdd(&out_color[index].y, finalColor.y);
	atomicAdd(&out_color[index].z, finalColor.z);
};

/*
Color World::shade_ray(Ray ray)
{
  if(ray.getLevel() > RECURSION_DEPTH) return background;
  firstIntersection(ray);
  if(ray.didHit())
  {
	// cout << ray.getDirection() << " " << ray.getIdx() << endl;
	// cerr << ray.getOrigin() << " " << ray.getDirection() << " " << ray.didHit() << endl;
	Color shadowColor(0.0,0.0,0.0);
	bool isShadow = false;
	//Run Shadow Test
	const Object* intersectedObject = ray.intersected();
	for(LightSource* ls : this->lightSourceList) {
	  Ray shadowRay(ray.getPosition()+0.01*(ls->getPosition()-ray.getPosition()),ls->getPosition()-ray.getPosition());
	  firstIntersection(shadowRay);
	  if(shadowRay.didHit()) {
		isShadow = true;
		shadowColor = ambient*(intersectedObject->getMaterial()->shade(ray))*(intersectedObject->getMaterial())->ka;
	  }
	}
	//..Compute Shade factor due to light
	Color lightColor(0.0,0.0,0.0);
	for(LightSource* ls : this->lightSourceList) {
	  // cerr << ray.getOrigin() << " " << ray.getDirection() << " " << ray.didHit() << " ";
	  // cerr << intersectedObject << endl;
	  lightColor = lightColor + get_light_shade(ray.getPosition(),intersectedObject->getNormalAtPosition(ray.getPosition()),*ls,intersectedObject->getMaterial(),ray.getDirection());
	}
	lightColor = lightColor + ambient*(intersectedObject->getMaterial()->shade(ray))*(intersectedObject->getMaterial())->ka;
	//if(shadowEffect) lightColor = lightColor*intersectedObject->getMaterial()->ka;

	Color finalColor = lightColor;
	if(isShadow) finalColor = finalColor*(1e-4) + shadowColor*(1 - 1e-4);

	//Reflection
	auto N = intersectedObject->getNormalAtPosition(ray.getPosition());
	auto I = ray.getDirection();
	N.normalize();
	I.normalize();

	double eta = intersectedObject->getMaterial()->eta;
	Vector3D T(0.0,0.0,0.0);
	double t = ray.getParameter();
	double c = 0;
	Vector3D k(1.0,1.0,1.0);
	int level = ray.getLevel();
	if(intersectedObject->getMaterial()->kr > 0 && intersectedObject->getMaterial()->kt > 0)
	{
	  //Dielectrics
	  auto R = reflect(I,N);
	  if(dotProduct(ray.getDirection(),N) < 0)
	  {
		refract(I,N,eta,T);
		c = -dotProduct(I,N);
	  }
	  else
	  {
		k = Vector3D(pow(EULER_CONSTANT,-1.0*0.27*t),pow(EULER_CONSTANT,-1.0*0.45*t),pow(EULER_CONSTANT,-1.0*0.55*t));
		if(refract(I,-1.0*N,1/eta,T)) c = dotProduct(T,N);
		else {
		  Ray temp = Ray(ray.getPosition()+ 1e-4 * R,R,level+1);
		  return k*shade_ray(temp);
		}
	  }
	  double _R0 = ((eta-1)*(eta-1))/((eta+1)*(eta+1));
	  double _R = _R0 + (1-_R0)*pow(1-c,5);
	  Ray temp1 = Ray(ray.getPosition()+ 1e-4 * R,R,level+1);
	  Ray temp2 = Ray(ray.getPosition()+ 1e-4 * T,T,level*2);
	  return k*(_R * shade_ray(temp1) + (1-_R)*shade_ray(temp2));
	}
	else if(intersectedObject->getMaterial()->kr > 0)
	{
	  auto R = reflect(I,N);
	  Ray reflectedRay(ray.getPosition()+ 1e-4 * R,R, level + 1);

	  finalColor = finalColor + (intersectedObject->getMaterial()->kr)*shade_ray(reflectedRay);
	}
	return finalColor;
  }
  return background;
}
*/
///////////////////////////////////////////////////////////////////
// Convert to RGBA kernel
// Parameters:
// color = Color array in floats
// d_out = Output array as RGBA unsigned char
// w = width
// h = height
//////////////////////////////////////////////////////////////////
__global__ void convert_to_rgba(float3 *color, uchar4* d_out, int w, int h)
{
	if (!color || !d_out) return ;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	int index = i + j * w; // 1D indexing
	get_color_from_float3(color[index], d_out + index);
}

int damnCeil(int num, int den) {
	return (num / den) + (num % den != 0);
}

__global__ void get_bounds(float * xmin, float * xmax, float * ymin, float * ymax, float * zmin, float * zmax, Triangle * triangles, int num_triangles) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < num_triangles) {
		triangles[idx].getWorldBound(xmin[idx], xmax[idx], ymin[idx], ymax[idx], zmin[idx], zmax[idx]);
	}
}

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ unsigned int morton3D(float x, float y, float z, BBox * bounds) {
	x = (x - bounds->axis_min[0]) / (bounds->axis_max[0] - bounds->axis_min[0]);
	y = (y - bounds->axis_min[1]) / (bounds->axis_max[1] - bounds->axis_min[1]);
	z = (z - bounds->axis_min[2]) / (bounds->axis_max[2] - bounds->axis_min[2]);
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

__global__ void generate_morton_codes(unsigned int * morton_codes, float * xmin, float * xmax, float * ymin,
                                float * ymax, float * zmin, float * zmax, BBox * bounds, int num_triangles) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < num_triangles) {
        morton_codes[idx] = morton3D(xmin[idx] + xmax[idx] / 2,
                                ymin[idx] + ymax[idx] / 2,
                                zmin[idx] + zmax[idx] / 2, bounds);
    }
}

int findSplit(unsigned int * sorted_codes, int first, int last) {
    // Identical Morton codes => split the range in the middle.

    unsigned int firstCode = sorted_codes[first];
    unsigned int lastCode = sorted_codes[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

//    int commonPrefix = __clz(firstCode ^ lastCode);
    int commonPrefix = __builtin_clz(firstCode ^ lastCode);
    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    int split = first; // initial guess
    int step = last - first;

    do {
        step = (step + 1) >> 1; // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last) {
            unsigned int splitCode = sorted_codes[newSplit];
//            int splitPrefix = __clz(firstCode ^ splitCode);
            int splitPrefix = __builtin_clz(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    } while (step > 1);

    return split;
}

void generateHierarchy(unsigned int * sorted_codes, int first, int last, Triangle * triangles, BVHTree& tree, int& idx) {
//	int curr_idx = idx;
//	if(first == last) {
//		tree.isLeaf[curr_idx] = true;
//		tree.primitive_idx[curr_idx] = first;
//		tree.left[curr_idx] = - 1;
//		tree.right[curr_idx] = -1;
//		tree.bbox[curr_idx] = triangles[first].getWorldBound();
//		return;
//    }
//
//	int split = findSplit(sorted_codes, first, last);
//    int left_idx = ++idx;
//    generateHierarchy(sorted_codes, first, split, triangles, tree, idx);
//    int right_idx = ++idx;
//    generateHierarchy(sorted_codes, split + 1, last, triangles, tree, idx);
//    tree.left[curr_idx] = left_idx, tree.right[curr_idx] = right_idx;
//    tree.isLeaf[curr_idx] = false;
//    tree.primitive_idx[curr_idx] = -1;
//    tree.bbox[curr_idx] = tree.bbox[left_idx];
//    tree.bbox[curr_idx].doUnion(tree.bbox[right_idx]);
//    return;

	queue < int > first_queue, last_queue, node_queue;
	node_queue.push(idx); idx++;
	first_queue.push(first);
	last_queue.push(last);
	while(!node_queue.empty()) {
		int node = node_queue.front(); node_queue.pop();
		int node_first = first_queue.front(); first_queue.pop();
		int node_last = last_queue.front(); last_queue.pop();
		if(node_first != node_last) {
			tree.isLeaf[node] = false;
			tree.primitive_idx[node] = -1;
			int split = findSplit(sorted_codes, node_first, node_last);
			tree.left[node] = idx;
			tree.right[node] = idx + 1;
			node_queue.push(idx); node_queue.push(idx + 1);
			idx += 2;
			first_queue.push(node_first); first_queue.push(split + 1);
			last_queue.push(split); last_queue.push(node_last);
		} else {
			tree.isLeaf[node] = true;
			tree.primitive_idx[node] = node_first;
			tree.left[node] = - 1;
			tree.right[node] = -1;
			tree.bbox[node] = triangles[node_first].getWorldBound();
		}
	}

	for(int i = idx - 1; i >= 0; i--) {
		if(!tree.isLeaf[i]) {
			tree.bbox[i] = tree.bbox[tree.left[i]];
			tree.bbox[i].doUnion(tree.bbox[tree.right[i]]);
		}
	}
}

void buildTree(int w, int h, Triangle * triangles, int num_triangles) {

    //checkCudaErrors(cudaMalloc((void**)&colors, sizeof(float3)*w*h));
    //checkCudaErrors(cudaMalloc((void**)&d_rays[0], sizeof(Ray)*w*h));

    float * xmin, * xmax, * ymin, * ymax, * zmin, * zmax;
    cudaMalloc(&xmin, sizeof(float) * num_triangles);
    cudaMalloc(&xmax, sizeof(float) * num_triangles);
    cudaMalloc(&ymin, sizeof(float) * num_triangles);
    cudaMalloc(&ymax, sizeof(float) * num_triangles);
    cudaMalloc(&zmin, sizeof(float) * num_triangles);
    cudaMalloc(&zmax, sizeof(float) * num_triangles);

    const dim3 blockSize(TX * TY);
    const dim3 gridSizeTriangles(damnCeil(num_triangles, TX * TY));

    get_bounds <<< gridSizeTriangles, blockSize >>> (xmin, xmax, ymin, ymax, zmin, zmax, triangles, num_triangles);

    thrust::tuple <float, float, float> axis_min, axis_max;

    thrust::device_ptr < float > xminptr = thrust::device_pointer_cast(xmin);
    thrust::device_ptr < float > xmaxptr = thrust::device_pointer_cast(xmax);
    thrust::device_ptr < float > yminptr = thrust::device_pointer_cast(ymin);
    thrust::device_ptr < float > ymaxptr = thrust::device_pointer_cast(ymax);
    thrust::device_ptr < float > zminptr = thrust::device_pointer_cast(zmin);
    thrust::device_ptr < float > zmaxptr = thrust::device_pointer_cast(zmax);

    BBox bounds, * d_bounds;
    checkCudaErrors(cudaMalloc(&d_bounds, sizeof(BBox)));
    bounds.axis_min[0] = thrust::reduce(xminptr, xminptr + num_triangles, 1e36, thrust::minimum<float>());
    bounds.axis_min[1] = thrust::reduce(yminptr, yminptr + num_triangles, 1e36, thrust::minimum<float>());
    bounds.axis_min[2] = thrust::reduce(zminptr, zminptr + num_triangles, 1e36, thrust::minimum<float>());
    bounds.axis_max[0] = thrust::reduce(xmaxptr, xmaxptr + num_triangles, -1e36, thrust::maximum<float>());
    bounds.axis_max[1] = thrust::reduce(ymaxptr, ymaxptr + num_triangles, -1e36, thrust::maximum<float>());
    bounds.axis_max[2] = thrust::reduce(zmaxptr, zmaxptr + num_triangles, -1e36, thrust::maximum<float>());
    cout << bounds.axis_min[0] << " " << bounds.axis_max[0] << endl;
	cout << bounds.axis_min[1] << " " << bounds.axis_max[1] << endl;
	cout << bounds.axis_min[2] << " " << bounds.axis_max[2] << endl;
    checkCudaErrors(cudaMemcpy(d_bounds, &bounds, sizeof(BBox), cudaMemcpyHostToDevice));
//    cudaDeviceSynchronize();
//    printf("%f %f\n", bounds.axis_min[0], bounds.axis_max[0]);
//    printf("%f %f\n", bounds.axis_min[1], bounds.axis_max[1]);
//    printf("%f %f\n", bounds.axis_min[2], bounds.axis_max[2]);
//    checkCudaErrors(cudaMemcpy(&bounds, d_bounds, sizeof(BBox), cudaMemcpyDeviceToHost));
//    printf("%f %f\n", bounds.axis_min[0], bounds.axis_max[0]);
//	printf("%f %f\n", bounds.axis_min[1], bounds.axis_max[1]);
//	printf("%f %f\n", bounds.axis_min[2], bounds.axis_max[2]);
    unsigned int * d_morton_codes;
    checkCudaErrors(cudaMalloc(&d_morton_codes, sizeof(unsigned int) * num_triangles));
    cudaDeviceSynchronize();
    cout << "allocation done" << endl;
    generate_morton_codes <<< gridSizeTriangles, blockSize >>> (d_morton_codes, xmin, xmax, ymin, ymax, zmin,
                                                                zmax, d_bounds, num_triangles);
    unsigned int * morton_codes = (unsigned int *) malloc(sizeof(unsigned int) * num_triangles);
    Triangle * h_triangles = (Triangle *) malloc(sizeof(Triangle) * num_triangles);

    checkCudaErrors(cudaMemcpy(morton_codes, d_morton_codes, sizeof(unsigned int) * num_triangles, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_triangles, triangles, sizeof(Triangle) * num_triangles, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    cout << "morton codes done" << endl;

    // thrust sort and stuff begin
    thrust::sort_by_key(morton_codes, morton_codes + num_triangles, h_triangles);
    checkCudaErrors(cudaMemcpy(triangles, h_triangles, sizeof(Triangle) * num_triangles, cudaMemcpyHostToDevice));
    // thrust sort and stuff done
    cudaDeviceSynchronize();
    cout << "sort stuff done" << endl;

    int idx = 0;
    BVHTree h_tree(3 * num_triangles), h_dtree_holder;

    generateHierarchy(morton_codes, 0, num_triangles - 1, h_triangles, h_tree, idx);
    cout << num_triangles << " "  << idx << " " << (float) idx / num_triangles << endl;
    checkCudaErrors(cudaMalloc(&h_dtree_holder.bbox, sizeof(BBox) * num_triangles * 3));
    checkCudaErrors(cudaMalloc(&h_dtree_holder.left, sizeof(int)  * num_triangles * 3));
    checkCudaErrors(cudaMalloc(&h_dtree_holder.right, sizeof(int) * num_triangles * 3));
    checkCudaErrors(cudaMalloc(&h_dtree_holder.primitive_idx, sizeof(int) * num_triangles * 3));
    checkCudaErrors(cudaMalloc(&h_dtree_holder.isLeaf, sizeof(bool) * num_triangles * 3));

    checkCudaErrors(cudaMemcpy(h_dtree_holder.bbox, h_tree.bbox, sizeof(BBox) * num_triangles * 3, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(h_dtree_holder.left, h_tree.left, sizeof(int) * num_triangles * 3, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(h_dtree_holder.right, h_tree.right, sizeof(int) * num_triangles * 3, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(h_dtree_holder.primitive_idx, h_tree.primitive_idx, sizeof(int) * num_triangles * 3, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(h_dtree_holder.isLeaf, h_tree.isLeaf, sizeof(bool) * num_triangles * 3, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&d_tree, sizeof(BVHTree)));
    checkCudaErrors(cudaMemcpy(d_tree, &h_dtree_holder, sizeof(BVHTree), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    cout << "all memory copy done" << endl;
    checkCudaErrors(cudaFree(xmin));
    checkCudaErrors(cudaFree(xmax));
    checkCudaErrors(cudaFree(ymin));
    checkCudaErrors(cudaFree(ymax));
    checkCudaErrors(cudaFree(zmin));
    checkCudaErrors(cudaFree(zmax));
}

void create_space_for_kernels(int w, int h)
{

	checkCudaErrors(cudaMalloc((void**)&colors, sizeof(float3)*w * h));
	//checkCudaErrors(cudaMalloc((void**)&d_rays[0], sizeof(Ray)*w*h));
	for (int i = 0; i < 7; i ++)
	{
		checkCudaErrors(cudaMalloc((void**)&d_rays[i], sizeof(Ray)*w * h));
		if (i) checkCudaErrors(cudaMalloc((void**)&d_coeffs[i], sizeof(float)*w * h));
	}

	cudaEventCreate(&event);
	cudaStreamCreate(&streamA1);
	cudaStreamCreate(&streamA2);
	cudaStreamCreate(&streamA3);
	cudaStreamCreate(&streamA4);

	d_coeffs[0] = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_d_coeffs, sizeof(float*) * 7));
	checkCudaErrors(cudaMemcpy(d_d_coeffs, d_coeffs, sizeof(float*) * 7, cudaMemcpyHostToDevice));
}

void free_space_for_kernels()
{
	//if(colors) checkCudaErrors(cudaFree(colors));
	for (int i = 0; i < 7; i ++)
	{
		checkCudaErrors(cudaFree(d_rays[i]));
		if (i && d_coeffs[i]) checkCudaErrors(cudaFree(d_coeffs[i]));
	}

	cudaStreamDestroy(streamA1);
	cudaStreamDestroy(streamA2);
	cudaStreamDestroy(streamA3);
	cudaStreamDestroy(streamA4);
	cudaEventDestroy(event);

	checkCudaErrors(cudaFree(d_d_coeffs));
}

void kernelLauncher(uchar4 *d_out, int w, int h, Camera* camera, Triangle* triangles, int num_triangles, LightSource* l) {
	const dim3 blockSize(TX, TY);
	const dim3 gridSize = dim3(w / TX, h / TY);

	//Start Procedure
	cudaProfilerStart();

	createRaysAndResetImage <<< gridSize, blockSize>>>(camera, w, h, d_rays[0], d_out, d_d_coeffs, colors);
	cudaDeviceSynchronize();

	//Karlo Ray trace 1000 baar yahaan
	//A
	raytrace <<< gridSize, blockSize, 0, streamA1>>>(colors, d_coeffs[0], w, h, d_rays[0], d_rays[1], d_coeffs[1], d_rays[2], d_coeffs[2], triangles, num_triangles, l, d_tree);
	//cudaEventRecord(event);
	cudaDeviceSynchronize();

	//Run these 2 concurrently
	//A1
	raytrace <<< gridSize, blockSize, 0, streamA1>>>(colors, d_coeffs[1], w, h, d_rays[1], d_rays[3], d_coeffs[3], d_rays[4], d_coeffs[4], triangles, num_triangles, l, d_tree);
	//A2
	raytrace <<< gridSize, blockSize, 0, streamA2>>>(colors, d_coeffs[2], w, h, d_rays[2], d_rays[5], d_coeffs[5], d_rays[6], d_coeffs[6], triangles, num_triangles, l, d_tree);
	//cudaEventRecord(event);
	cudaDeviceSynchronize();

	//Run these 4 concurrently
	//A11
	raytrace <<< gridSize, blockSize, 0, streamA1>>>(colors, d_coeffs[3], w, h, d_rays[3], NULL, NULL, NULL, NULL, triangles, num_triangles, l, d_tree);
	//A12
	raytrace <<< gridSize, blockSize, 0, streamA2>>>(colors, d_coeffs[4], w, h, d_rays[4], NULL, NULL, NULL, NULL, triangles, num_triangles, l, d_tree);
	//A21
	raytrace <<< gridSize, blockSize, 0, streamA3>>>(colors, d_coeffs[5], w, h, d_rays[5], NULL, NULL, NULL, NULL, triangles, num_triangles, l, d_tree);
	//A22
	raytrace <<< gridSize, blockSize, 0, streamA4>>>(colors, d_coeffs[6], w, h, d_rays[6], NULL, NULL, NULL, NULL, triangles, num_triangles, l, d_tree);
	//cudaEventRecord(event);
	cudaDeviceSynchronize();

	//Final Output Array
	convert_to_rgba <<< gridSize, blockSize>>>(colors, d_out, w, h);
	cudaDeviceSynchronize();
	cudaProfilerStop();
}
