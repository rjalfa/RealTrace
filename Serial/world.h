#ifndef _WORLD_H_
#define _WORLD_H_

#include <vector>
#include "object.h"
#include "lightsource.h"
#include "color.h"
#include "ray.h"
#include "uniform-grid.h"

#define RECURSION_DEPTH 3
class World
{
private:
	std::vector<Object*> objectList;
	std::vector<LightSource*> lightSourceList;
	Color ambient;
	Color background; //Background color to shade rays that miss all objects
	Color get_light_shade(const Vector3D& position, const Vector3D& normal, const LightSource& lightSource,const Material* mat, const Vector3D viewVector);

public:
	UniformGrid uniform_grid;
	World():
		objectList(0), lightSourceList(0), ambient(0), background(0)
	{}
	void setBackground(const Color& bk) { background = bk;}
	Color getbackground() { return background;}
	void setAmbient(const Color& amb) {ambient = amb;}
	Color getAmbient() {return ambient;}
	void addLight(LightSource* ls)
	{
		lightSourceList.push_back(ls);
	}
	void addObject(Object *obj)
	{
		objectList.push_back(obj);
	}
	
	vector < Object * >& getObjectList() {
		return objectList;
	}

	float firstIntersection(Ray& ray);
	Color shade_ray(Ray ray);
};
#endif
