//plane.h
#ifndef _PLANE_H_
#define _PLANE_H_

#include "object.h"
#include "ray.h"
#include "vector3D.h"
#include "color.h"

class Plane : public Object
{
private:
	Vector3D position1;
	Vector3D position2;
	Vector3D position3;
	Vector3D position4;
	Vector3D Normal;
	bool pointInside(const Vector3D point) const;
	bool triangleIntersect(Ray& r,Vector3D vertexA,Vector3D vertexB,Vector3D vertexC) const;
public:
	Plane(const Vector3D& _pos1,const Vector3D& _pos2,const Vector3D& _pos3,const Vector3D& _pos4, Material* mat):
		Object(mat), position1(_pos1),position2(_pos2),position3(_pos3),position4(_pos4)
	{
		Normal = crossProduct(_pos3-_pos1,_pos2-_pos1);
		isSolid = true;
	}
	
	bool intersect(Ray& r) const;
	Vector3D getNormalAtPosition(const Vector3D& position) const; 
};
#endif
