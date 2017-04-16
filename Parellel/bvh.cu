#include "structures.h"
#include <thrust/sort.h>

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
unsigned int morton3D(float x, float y, float z, BBox * bounds) {
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

    unsigned int firstCode = sortedMortonCodes[first];
    unsigned int lastCode = sortedMortonCodes[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    int commonPrefix = __clz(firstCode ^ lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    int split = first; // initial guess
    int step = last - first;

    do {
        step = (step + 1) >> 1; // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last) {
            unsigned int splitCode = sortedMortonCodes[newSplit];
            int splitPrefix = __clz(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    } while (step > 1);

    return split;
}

BVH_Node * generateHierarchy(unsigned int * sorted_codes, int first, int last, Triangle * triangles) {
    if(first == last) {
        BVH_Node * temp = new BVH_Node(NULL, NULL, first, last);
        temp->bbox = triangles[first].getWorldBound();
        return temp;
    }

    int split = findSplit(sorted_codes, first, last);

    BVH_Node * temp = new BVH_Node();
    temp->left = generateHierarchy(sorted_codes, first, split, triangles);
    temp->right = generateHierarchy(sorted_codes, split + 1, last, triangles);
    temp->calcBBox();
    return temp;
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

    UniformGrid h_uniform_grid;

    BBox bounds, * d_bounds;
    checkCudaErrors(cudaMalloc(&d_bounds, sizeof(BBox)));
    checkCudaErrors(cudaMemcpy(d_bounds, &bounds, sizeof(BBox), cudaMemcpyHostToDevice));
    bounds.axis_min[0] = thrust::reduce(xminptr, xminptr + num_triangles, 1e36, thrust::minimum<float>());
    bounds.axis_min[1] = thrust::reduce(yminptr, yminptr + num_triangles, 1e36, thrust::minimum<float>());
    bounds.axis_min[2] = thrust::reduce(zminptr, zminptr + num_triangles, 1e36, thrust::minimum<float>());
    bounds.axis_max[0] = thrust::reduce(xmaxptr, xmaxptr + num_triangles, -1e36, thrust::maximum<float>());
    bounds.axis_max[1] = thrust::reduce(ymaxptr, ymaxptr + num_triangles, -1e36, thrust::maximum<float>());
    bounds.axis_max[2] = thrust::reduce(zmaxptr, zmaxptr + num_triangles, -1e36, thrust::maximum<float>());

    unsigned int * d_morton_codes;
    checkCudaErrors(cudaMalloc(&d_morton_codes, sizeof(unsigned int) * num_triangles));

    generate_morton_codes <<< gridSizeTriangles, blockSize >>> (d_morton_codes, xmin, xmax, ymin, ymax, zmin, 
                                                                zmax, d_bounds, num_triangles);

    // thrust sort and stuff begin
    thrust::sort_by_key(d_morton_codes, d_morton_codes + num_triangles, triangles);
    Triangle * h_triangles = (Triangle *) malloc(sizeof(Triangle) * num_triangles);
    // thrust sort and stuff done

    unsigned int * morton_codes = (unsigned int *) malloc(sizeof(unsigned int) * num_triangles);

    checkCudaErrors(cudaMemcpy(morton_codes, d_morton_codes, sizeof(unsigned int) * num_triangles, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_triangles, triangles, sizeof(Triangle) * num_triangles, cudaMemcpyDeviceToHost));
    generateHierarchy(morton_codes, 0, num_triangles - 1, h_triangles);

    checkCudaErrors(cudaFree(xmin));
    checkCudaErrors(cudaFree(xmax));
    checkCudaErrors(cudaFree(ymin));
    checkCudaErrors(cudaFree(ymax));
    checkCudaErrors(cudaFree(zmin));
    checkCudaErrors(cudaFree(zmax));
}
