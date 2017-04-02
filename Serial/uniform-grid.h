#ifndef _UNIFORM_GRID_H_
#define _UNIFORM_GRID_H_

#include <vector>
#include "triangle.h"
#include "ray.h"
#include "utilities.h"
#include "vector3D.h"

using namespace std;

class Voxel {
public:
	vector < int > idx;
	vector < Triangle * > primitives;
	void addPrimitive(Triangle * p, int i);

	bool intersect(Ray& ray);
};


class UniformGrid {

private:
	BBox bounds;
	vector < double > delta;
	vector < int > nVoxels;
	double voxelsPerUnitDist;
	vector < double > width, invWidth;
	// Voxel ** voxels;
	vector < Voxel > voxels;
	int nv;
	double findVoxelsPerUnitDist(vector < double > delta, int num);

	int posToVoxel(const Vector3D& pos, int axis);

	float voxelToPos(int p, int axis);

	inline int offset(double x, double y, double z);

public:
	UniformGrid() {}
	UniformGrid(vector < Triangle * > &p);
	// void free() {
	// 	for(int i = 0; i < nv; i++)
	// 		if(voxels[i] != NULL) delete voxels[i];
	// }
	bool intersect(Ray& ray);
};

#endif