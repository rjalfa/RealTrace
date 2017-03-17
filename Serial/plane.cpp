//plane.cpp

#include "plane.h"
#include "utilities.h"
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cassert>
#define EPSILON 1e-7
using namespace std;

bool Plane::triangleIntersect(Ray& r,Vector3D vertexA,Vector3D vertexB,Vector3D vertexC) const
{
	//barycentric coordinate check
	double A = determinant(vertexA-vertexB,vertexA-vertexC,r.getDirection());
	if(std::abs(A) < EPSILON) return false;
	double beta = determinant(vertexA-r.getOrigin(),vertexA-vertexC,r.getDirection()) / A;
	double gamma = determinant(vertexA-vertexB,vertexA-r.getOrigin(),r.getDirection()) / A;
	double t = determinant(vertexA-vertexB,vertexA-vertexC,vertexA-r.getOrigin()) / A;
	if(beta > 0.0 && gamma > 0.0 && beta+gamma < 1.0) {r.setParameter(t,this); return true;}
	return false;
}

bool Plane::intersect(Ray& r) const
{
	return triangleIntersect(r,position1,position2,position3) || triangleIntersect(r,position1,position3,position4);
}

Vector3D Plane::getNormalAtPosition(const Vector3D& position) const
{
	return this->Normal;
}

