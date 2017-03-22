#ifndef _CYLINDER_H_
#define _CYLINDER_H_

#include <vector>
#include "triangle.h"
#include "ray.h"
#include "utilities.h"
#include "Vector3D.h"

using namespace std;

class Voxel {
public:
	vector < Triangle > primitives;
	void addPrimitive(Triangle& p);

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
	UniformGrid(vector < Triangle > &p);
	// void free() {
	// 	for(int i = 0; i < nv; i++)
	// 		if(voxels[i] != NULL) delete voxels[i];
	// }
	bool intersect(Ray& ray);
};

#endif