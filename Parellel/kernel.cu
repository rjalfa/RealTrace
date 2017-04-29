#include "kernel.h"
#include "structures.h"
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/scan.h>
#include "camera.h"
#include "helper_cuda.h"
#include "cuda_profiler_api.h"
#include "utilities.h"
#define AMBIENT_COLOR make_float3(0.235294, 0.67451, 0.843137)
#define KR 0.3
#define KT 0.3
#define EULER_CONSTANT 2.718
#define ETA 1.5
#define KA 0.4

__device__ inline unsigned char clip(float x) { return x > 255 ? 255 : (x < 0 ? 0 : x); }

UniformGrid * d_uniform_grid;
float3* colors = 0;

Ray* d_rays[7];
float* d_coeffs[7];
float** d_d_coeffs = NULL;
const int MAXSTREAMS = 4;
cudaEvent_t event[MAXSTREAMS];
cudaStream_t stream[MAXSTREAMS];

// kernel function to compute decay and shading
__device__ inline void get_color_from_float3(float3 color, uchar4* cell)
{
	cell->x = clip(color.x * 255);
	cell->y = clip(color.y * 255);
	cell->z = clip(color.z * 255);
	cell->w = 255;
}

__device__ float3 refract(const float3 &I, const float3 &N, const float &ior) 
{ 
	float cosi = clamp(-1, 1, dotProduct(I, N)); 
	float etai = 1, etat = ior; 
	float3 n = N; 
	if (cosi < 0) { cosi = -cosi; } else { 
		float temp = etai;
		etai = etat;
		etat = temp;
		//swap(etai, etat); 
		n= -N;
	} 
	float eta = etai / etat; 
	float k = 1 - eta * eta * (1 - cosi * cosi); 
	return (k < 0) ? make_float3(0,0,0) : (eta * I + (eta * cosi - __fsqrt_rz(k)) * n); 
}

__device__ __forceinline__ void fresnel(const float3& I, const float3& N, const float& ior, float &kr)
{
	float cosi = clamp(-1, 1, dotProduct(I, N));
	float etai = 1, etat = ior;
	if (cosi > 0) {
		float t = etai;
		etai = etat;
		etat = t;
	}
	// Compute sini using Snell's law
	float sint = etai / etat * __fsqrt_rz(max(0.f, 1 - cosi * cosi));
	// Total internal reflection
	if (sint >= 1) {
		kr = 1;
	}
	else {
		float cost = __fsqrt_rz(fmaxf(0.f, 1 - sint * sint));
		cosi = fabsf(cosi);
		float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
		float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
		kr = (Rs * Rs + Rp * Rp) / 2;
	}
}

//Uniform Grid Intersect
//__device__ void getFirstIntersection(UniformGrid * ug, Ray& r) {
//  ug->intersect(r);
//}

//Global Memory loop intersect
/*
__device__ void intersect(Triangle* triangles, int num_triangles, Ray* r)
{
  for(int i = 0; i < num_triangles; i ++) triangles[i].intersect(r);
}
*/

//Shared Memory Loop Intersect

__device__ inline void intersect(Triangle* triangles, int num_triangles, Ray* r, UniformGrid * ug, float in_coeff)
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
	ug->intersect(triangles, *r, in_coeff);
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
	for(int i = 0; i < 7; i ++) if(d_coeffs[i] != NULL) d_coeffs[i][index] = 0;
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
__global__ void raytrace(float3 *out_color, float* in_coeffs, int w, int h, Ray* rays, Ray* out_rays_reflect, float* out_coeffs_reflect, Ray* out_rays_refract, float* out_coeffs_refract, Triangle* triangles, int num_triangles, LightSource* l, UniformGrid * ug)
{
	if (out_color == NULL || rays == NULL) return;
	//int i = blockDim.x * blockIdx.x + threadIdx.x;
	//int j = blockDim.y * blockIdx.y + threadIdx.y;
	int index = (blockDim.x * blockIdx.x + threadIdx.x) + (blockDim.y * blockIdx.y + threadIdx.y) * w;
	//Switches
	float in_coeff = ((in_coeffs != NULL) ? in_coeffs[index] : 1.00);
	in_coeff = __saturatef(in_coeff);
	//clamp(in_coeff, 0, 1);
	Ray ray = rays[index];

	int flag = 0;
	flag |= (out_rays_refract != NULL && out_coeffs_refract != NULL);
	flag |= (out_rays_reflect != NULL && out_coeffs_reflect != NULL) << 1;
	if(out_coeffs_reflect != NULL) out_coeffs_reflect[index] = 0;
	if(out_coeffs_refract != NULL) out_coeffs_refract[index] = 0;

	//Get owned ray
	
	intersect(triangles, num_triangles, &ray, ug, in_coeff);
	if (in_coeff < EPSILON || ray.direction == make_float3(0, 0, 0)) return;
	//bool reflect_over_refract = false;
	//Do one time intersection
	float3 finalColor = AMBIENT_COLOR;
	Triangle* intersected = 0;
	if (ray.has_intersected) {
		intersected = ray.intersected;
		float3 I = normalize(ray.direction);
		float3 N = normalize(intersected->get_normal());
		flag |= (intersected->type_of_material == REFLECTIVE) << 2;
		flag |= (intersected->type_of_material == REFRACTIVE) << 3;
		finalColor = get_light_color(get_point(&ray, ray.t), N, l, intersected, I);
		finalColor += (intersected)->color * KA;
		//Reflect
		float kr;
		bool outside = (dotProduct(I, N) < 0);
		float eff_eta = ETA;
		fresnel(I, N, eff_eta, kr);
		float3 bias = N * 1e-4f;
		float3 hitPoint = ray.getPosition();
		
		if(flag & 4) kr = KR;

		if(flag & 6 == 6 || ((flag & 1) && (flag & 8)))
		{
			float3 reflectionDirection = reflect(I, N);
			float3 reflectionRayOrig = outside ? hitPoint + bias : hitPoint - bias;
			out_rays_reflect[index] = Ray(reflectionRayOrig, reflectionDirection);
			out_coeffs_reflect[index] = in_coeff * kr;
			in_coeff *= (1 - kr);
		}
		
		if ((flag & 1) && (flag & 8))
		{
			if (kr < 1) {
				float3 refractionDirection = refract(I, N, eff_eta);
				float3 refractionRayOrig = outside ? hitPoint - bias : hitPoint + bias;
				out_rays_refract[index] = Ray(refractionRayOrig, refractionDirection);
				out_coeffs_refract[index] = in_coeff * (1 - kr);
			}
			in_coeff = 0.0;
		}
	}

	finalColor *= in_coeff;
	atomicAdd(&out_color[index].x, finalColor.x);
	atomicAdd(&out_color[index].y, finalColor.y);
	atomicAdd(&out_color[index].z, finalColor.z);
};

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

__global__ void count_sizes(UniformGrid * ug, Triangle * triangles, int num_triangles) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < num_triangles) {
		float xmin, xmax, ymin, ymax, zmin, zmax;
		triangles[idx].getWorldBound(xmin, xmax, ymin, ymax, zmin, zmax);

		int vxmin, vxmax, vymin, vymax, vzmin, vzmax;

		vxmin = ug->posToVoxel(xmin, 0);
		vxmax = ug->posToVoxel(xmax, 0);
		vymin = ug->posToVoxel(ymin, 1);
		vymax = ug->posToVoxel(ymax, 1);
		vzmin = ug->posToVoxel(zmin, 2);
		vzmax = ug->posToVoxel(zmax, 2);

		for (int z = vzmin; z <= vzmax; z++) {
			for (int y = vymin; y <= vymax; y++) {
				for (int x = vxmin; x <= vxmax; x++) {
					int o = ug->offset(x, y, z);
					atomicAdd(&(ug->lower_limit[o]), 1);
				}
			}
		}
	}
}

//__global__ void reserve_space(UniformGrid * ug, int nv) {
//  int idx = blockDim.x * blockIdx.x + threadIdx.x;
//  if(idx < nv) {
//    if(idx > 0) ug->voxels[idx].offset = ug->voxel_sizes[idx - 1];
//    else ug->voxels[idx].offset = 0;
//    if(idx > 0) ug->voxels[idx].max_size = ug->voxel_sizes[idx] - ug->voxel_sizes[idx - 1];
//    else ug->voxels[idx].max_size = ug->voxel_sizes[idx];
//    ug->voxels[idx].curr_size = 0;
////    printf("%d\n", ug->voxels[idx].max_size);
//  }
//}

__global__ void build_grid(UniformGrid * ug, Triangle * triangles, int num_triangles) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < num_triangles) {
		float xmin, xmax, ymin, ymax, zmin, zmax;
		triangles[idx].getWorldBound(xmin, xmax, ymin, ymax, zmin, zmax);

		int vxmin, vxmax, vymin, vymax, vzmin, vzmax;

		vxmin = ug->posToVoxel(xmin, 0);
		vxmax = ug->posToVoxel(xmax, 0);
		vymin = ug->posToVoxel(ymin, 1);
		vymax = ug->posToVoxel(ymax, 1);
		vzmin = ug->posToVoxel(zmin, 2);
		vzmax = ug->posToVoxel(zmax, 2);

		for (int z = vzmin; z <= vzmax; z++) {
			for (int y = vymin; y <= vymax; y++) {
				for (int x = vxmin; x <= vxmax; x++) {
					int o = ug->offset(x, y, z);
					Voxel::addPrimitive(ug, idx, o);
				}
			}
		}
	}
}

class justMax {
public:
	__host__ __device__
	thrust::tuple<float, float, float> operator()(thrust::tuple<float, float, float> a, thrust::tuple<float, float, float> b) {
		return thrust::make_tuple(max(thrust::get < 0 > (a), thrust::get < 0 > (b)), max(thrust::get < 1 > (a), thrust::get < 1 > (b)), max(thrust::get < 2 > (a), thrust::get < 2 > (b)));
	}
};

class justMin {
public:
	__host__ __device__
	thrust::tuple<float, float, float> operator()(thrust::tuple<float, float, float> a, thrust::tuple<float, float, float> b) {
		return thrust::make_tuple(min(thrust::get < 0 > (a), thrust::get < 0 > (b)), min(thrust::get < 1 > (a), thrust::get < 1 > (b)), min(thrust::get < 2 > (a), thrust::get < 2 > (b)));
	}
};

void buildGrid(int w, int h, Triangle * triangles, int num_triangles) {

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

	UniformGrid h_uniform_grid;

	h_uniform_grid.bounds.axis_min[0] = thrust::reduce(xminptr, xminptr + num_triangles, 1e36, thrust::minimum<float>());
	h_uniform_grid.bounds.axis_min[1] = thrust::reduce(yminptr, yminptr + num_triangles, 1e36, thrust::minimum<float>());
	h_uniform_grid.bounds.axis_min[2] = thrust::reduce(zminptr, zminptr + num_triangles, 1e36, thrust::minimum<float>());
	h_uniform_grid.bounds.axis_max[0] = thrust::reduce(xmaxptr, xmaxptr + num_triangles, -1e36, thrust::maximum<float>());
	h_uniform_grid.bounds.axis_max[1] = thrust::reduce(ymaxptr, ymaxptr + num_triangles, -1e36, thrust::maximum<float>());
	h_uniform_grid.bounds.axis_max[2] = thrust::reduce(zmaxptr, zmaxptr + num_triangles, -1e36, thrust::maximum<float>());

	h_uniform_grid.initialize(num_triangles);

	checkCudaErrors(cudaMalloc(&d_uniform_grid, sizeof(UniformGrid)));
	checkCudaErrors(cudaMemcpy(d_uniform_grid, &h_uniform_grid, sizeof(UniformGrid), cudaMemcpyHostToDevice));

	const dim3 gridSizeVoxels(damnCeil(h_uniform_grid.nv, TX * TY));
	count_sizes <<< gridSizeTriangles, blockSize >>> (d_uniform_grid, triangles, num_triangles);

	checkCudaErrors(cudaMemcpy(&h_uniform_grid, d_uniform_grid, sizeof(UniformGrid), cudaMemcpyDeviceToHost));

	thrust::device_ptr < int > voxel_sizes = thrust::device_pointer_cast(h_uniform_grid.lower_limit);
	int total_space = thrust::reduce(voxel_sizes, voxel_sizes + h_uniform_grid.nv);
	checkCudaErrors(cudaMalloc(&(h_uniform_grid.index_pool), sizeof(int) * total_space));
	thrust::exclusive_scan(voxel_sizes, voxel_sizes + h_uniform_grid.nv, voxel_sizes);

	checkCudaErrors(cudaMemcpy(h_uniform_grid.upper_limit, h_uniform_grid.lower_limit,
							   sizeof(int) * h_uniform_grid.nv, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_uniform_grid, &h_uniform_grid, sizeof(UniformGrid), cudaMemcpyHostToDevice));

//  reserve_space <<< gridSizeVoxels, blockSize >>> (d_uniform_grid, h_uniform_grid.nv);

	build_grid <<< gridSizeTriangles, blockSize >>> (d_uniform_grid, triangles, num_triangles);

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
	for (int i = 0; i < 7; i ++) {
		checkCudaErrors(cudaMalloc((void**)&d_rays[i], sizeof(Ray)*w * h));
		if(i) checkCudaErrors(cudaMalloc((void**)&d_coeffs[i], sizeof(float)*w * h));
	}
	for(int i = 0; i < MAXSTREAMS; i++) {
		cudaEventCreate(&event[i]);
		cudaStreamCreate(&stream[i]);
	}

	d_coeffs[0] = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_d_coeffs, sizeof(float*) * 7));
	checkCudaErrors(cudaMemcpy(d_d_coeffs, d_coeffs, sizeof(float*) * 7, cudaMemcpyHostToDevice));
}

void free_space_for_kernels()
{
	if(colors) checkCudaErrors(cudaFree(colors));
	for (int i = 0; i < 7; i ++) {
		checkCudaErrors(cudaFree(d_rays[i]));
		if(i && d_coeffs[i]) checkCudaErrors(cudaFree(d_coeffs[i]));
	}
	for(int i = 0; i < MAXSTREAMS; i++) {
		cudaEventDestroy(event[i]);
		cudaStreamDestroy(stream[i]);
	}
	
	checkCudaErrors(cudaFree(d_d_coeffs));
}

void kernelLauncher(uchar4 *d_out, int w, int h, Camera* camera, Triangle* triangles, int num_triangles, LightSource* l) {
	const dim3 blockSize(TX, TY);
	const dim3 gridSize = dim3(w / TX, h / TY);

	//Start Procedure
	// cudaProfilerStart();

	createRaysAndResetImage <<< gridSize, blockSize>>>(camera, w, h, d_rays[0], d_out, d_d_coeffs, colors);
	// cudaDeviceSynchronize();

	//Karlo Ray trace 1000 baar yahaan
	//A
	raytrace <<< gridSize, blockSize, 0, stream[0]>>>(colors, d_coeffs[0], w, h, d_rays[0], d_rays[1], d_coeffs[1], d_rays[2], d_coeffs[2], triangles, num_triangles, l, d_uniform_grid);
	cudaEventRecord(event[0], stream[0]);
	//Run these 2 concurrently
	
	//A1
	raytrace <<< gridSize, blockSize, 0, stream[0]>>>(colors, d_coeffs[1], w, h, d_rays[1], d_rays[3], d_coeffs[3], d_rays[4], d_coeffs[4], triangles, num_triangles, l, d_uniform_grid);
	//A2
	cudaStreamWaitEvent(stream[1], event[0], 0);
	raytrace <<< gridSize, blockSize, 0, stream[1]>>>(colors, d_coeffs[2], w, h, d_rays[2], d_rays[5], d_coeffs[5], d_rays[6], d_coeffs[6], triangles, num_triangles, l, d_uniform_grid);
	cudaEventRecord(event[1], stream[0]);
	cudaEventRecord(event[2], stream[1]);
	
	//Run these 4 concurrently
	//A11
	raytrace <<< gridSize, blockSize, 0, stream[0]>>>(colors, d_coeffs[3], w, h, d_rays[3], NULL, NULL, NULL, NULL, triangles, num_triangles, l, d_uniform_grid);
	//A12
	cudaStreamWaitEvent(stream[2], event[1], 0);
	raytrace <<< gridSize, blockSize, 0, stream[2]>>>(colors, d_coeffs[4], w, h, d_rays[4], NULL, NULL, NULL, NULL, triangles, num_triangles, l, d_uniform_grid);
	//A21
	raytrace <<< gridSize, blockSize, 0, stream[1]>>>(colors, d_coeffs[5], w, h, d_rays[5], NULL, NULL, NULL, NULL, triangles, num_triangles, l, d_uniform_grid);
	//A22
	cudaStreamWaitEvent(stream[3], event[2], 0);
	raytrace <<< gridSize, blockSize, 0, stream[3]>>>(colors, d_coeffs[6], w, h, d_rays[6], NULL, NULL, NULL, NULL, triangles, num_triangles, l, d_uniform_grid);

	//Final Output Array
	convert_to_rgba <<< gridSize, blockSize>>>(colors, d_out, w, h);
	cudaEventSynchronize(event[1]);
	cudaEventSynchronize(event[2]);
	// cudaProfilerStop();
}
