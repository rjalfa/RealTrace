//triangle.cpp

#include "triangle.h"
#include "utilities.h"
#include <cstdlib>
#include <iostream>
#include <cstdio>
using namespace std;

bool Triangle::intersect(Ray& r) const
{
	//barycentric coordinate check
	double A = determinant(vertexA-vertexB,vertexA-vertexC,r.getDirection());
	if(abs(A) < EPSILON) return false;
	double beta = determinant(vertexA-r.getOrigin(),vertexA-vertexC,r.getDirection()) / A;
	double gamma = determinant(vertexA-vertexB,vertexA-r.getOrigin(),r.getDirection()) / A;
	double t = determinant(vertexA-vertexB,vertexA-vertexC,vertexA-r.getOrigin()) / A;
	if(beta > 0.0 && gamma > 0.0 && beta+gamma < 1.0) {
		// cout << "setting" << endl;
		r.setParameter(t,this); return true;
	}
	return false;
}

Vector3D Triangle::getNormalAtPosition(const Vector3D& position) const
{
	return crossProduct(vertexA-vertexB,vertexA-vertexC);
}

BBox Triangle::getWorldBound() {
	BBox temp;
	for(int axis = 0; axis < 3; axis++) {
		for(int vno = 0; vno < 3; vno++) {
			temp.axis_min[axis] = min(temp.axis_min[axis], getVertex(vno).e[axis]);
			temp.axis_max[axis] = max(temp.axis_max[axis], getVertex(vno).e[axis]);
		}
	}
	return temp;
}