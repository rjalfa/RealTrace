//cylinder.h
#ifndef _CYLINDER_H_
#define _CYLINDER_H_

#include "object.h"
#include "ray.h"
#include "vector3D.h"
#include "color.h"

class Cylinder : public Object
{
private:
	Vector3D position;
	double radius;
	Vector3D up;
public:
	Cylinder(const Vector3D& _pos, double _rad,const Vector3D& u,Material* mat):
		Object(mat), position(_pos), radius(_rad), up(u)
	{
		isSolid = true;
	}
	
	bool intersect(Ray& r) const;
	Vector3D getNormalAtPosition(const Vector3D& position) const;
};
#endif
