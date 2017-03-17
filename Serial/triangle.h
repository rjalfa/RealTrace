//triangle.h
#ifndef _TRIANGLE_H_
#define _TRIANGLE_H_

#include "object.h"
#include "ray.h"
#include "vector3D.h"
#include "color.h"

class Triangle : public Object
{
private:
	Vector3D vertexA;
	Vector3D vertexB;
	Vector3D vertexC;

public:
		Triangle(const Vector3D& _pos1,const Vector3D& _pos2,const Vector3D& _pos3, Material* mat):
			Object(mat), vertexA(_pos1),vertexB(_pos2),vertexC(_pos3)
	{
		isSolid = true;
	}
	
	bool intersect(Ray& r) const;
	Vector3D getNormalAtPosition(const Vector3D& position) const;
};
#endif
