//matrial.h
#ifndef _MATERIAL_H_
#define _MATERIAL_H_

#include "color.h"
#include "ray.h"
#include <vector>
using namespace std;
class World;
class Triangle;
class Material
{
protected:
	World *world;
public:
	//Data

	Color color;
	double ka;//Ambient contribution
	double kd;//Diffuse constant
	double ks;//Specular constant
	double kr;//Contribution from reflection
	double kt;//Contribution from refraction
	double eta;//Coefficient of refraction
	double n;//Phong's shiny constant

	Material(World *w):
		world(w), color(0),
		ka(0.2), kd(1.0), ks(0.4), kr(0), kt(0),n(0), eta(128) {}
	virtual Color shade(const Ray& incident, const bool isSolid = true) const;

};

//Needs to be tied to a Triangle Object
class BarycentricMaterial : public Material
{
protected:
	Vector3D vertexA;
	Vector3D vertexB;
	Vector3D vertexC;
	vector<Color> colors;
public:
	//Data
	BarycentricMaterial(World *w, const Vector3D v1,const Vector3D v2,const Vector3D v3, const Color& c1, const Color& c2, const Color& c3): Material(w), vertexA(v1), vertexB(v2), vertexC(v3) {
		colors.push_back(c1);
		colors.push_back(c2);
		colors.push_back(c3);
	} 
	Color shade(const Ray& incident, const bool isSolid = true) const;
};
#endif
