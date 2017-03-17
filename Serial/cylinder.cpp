//cylinder.cpp

#include "cylinder.h"
bool solveQuadratic(double A,double B,double C,double &r1, double & r2)
{
	double D = B*B - 4*A*C;
	if(D < 0) return false;
	r1 = (-B + sqrt(D)) / (2*A);
	r2 = (-B - sqrt(D)) / (2*A);
	if(r1 > r2) swap(r1,r2);
	return true;
}

bool Cylinder::intersect(Ray& r) const
{
	//http://mrl.nyu.edu/~dzorin/rend05/lecture2.pdf
	//r, pa, va
	//p, v
	Vector3D temp1 = r.getDirection() - dotProduct(r.getDirection(),up)*up;
	Vector3D temp2 = r.getOrigin()-position - dotProduct(r.getOrigin()-position,up)*up;
	double A = dotProduct(temp1,temp1);
	double B = 2 * dotProduct(temp1,temp2);
	double C = dotProduct(temp2,temp2) - radius*radius;
	double t1,t2;
	if(solveQuadratic(A,B,C,t1,t2))
	{
		if(t1 > 0) r.setParameter(t1,this);
		else r.setParameter(t2,this);
		return true;
	}
	return false;
}

Vector3D Cylinder::getNormalAtPosition(const Vector3D& p) const
{
	double t = dotProduct(p-position,up)/dotProduct(up,up);
	return p - position - t*up;
}