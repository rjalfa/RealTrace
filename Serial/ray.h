//ray.h
#ifndef _RAY_H_
#define _RAY_H_

#include <float.h>
#include "vector3D.h"

class Object;

const float SMALLEST_DIST = 1e-4; //Constant used to dismiss intersections very close to previous
class Ray
{
private:
	const Vector3D origin;
	Vector3D direction;
	float t; //Distance travelled alogn the Ray
	bool hit; //has the ray hit something?
	int idx;
	const Object *object;//The object that has been hit
	int level;//Number of times the ray has been traced recursively
	float refractive_index;
	Vector3D normal; //Normal of the hit object

public:  
	Ray(const Vector3D& o, const Vector3D& d, int _level = 0, float _ref_idx = 1.0):
    		origin(o), direction(d), t(FLT_MAX), hit(false), level(_level), refractive_index(_ref_idx), object(nullptr)
	{
		direction.normalize();	
	}
	Vector3D getOrigin() const  {return origin;}
	Vector3D getDirection() const  {return direction;}
	Vector3D getPosition() const {return origin + t*direction;}
	Vector3D getNormal() const;
	void setNormal(const Vector3D norm) {this->normal = norm;} 
	float getParameter() const {return t;}
	void strictSetParameter(const float par) {
		t = par;
	}
	bool setParameter(const float par, const Object *obj);
	bool didHit() const {return hit;}
	void setHit(bool flag) {hit = flag;}
	void setIdx(int i) {idx = i;}
	int getIdx() {return idx;}
	const Object* intersected() const {return object;}
	int getLevel() const {return level;}
	
};
#endif
