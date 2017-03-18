#include "world.h"
#include "material.h"
#include "triangle.h"
#include "utilities.h"
Color Material::shade(const Ray& incident, const bool isSolid) const
{
	return color;
}

Color BarycentricMaterial::shade(const Ray& incident, const bool isSolid) const
{
	//Calculate Barycentric coords
	double A = determinant(vertexA-vertexB,vertexA-vertexC,incident.getDirection());
	if(abs(A) < EPSILON) return false;
	double beta = determinant(vertexA-incident.getOrigin(),vertexA-vertexC,incident.getDirection()) / A;
	double gamma = determinant(vertexA-vertexB,vertexA-incident.getOrigin(),incident.getDirection()) / A;
	double t = determinant(vertexA-vertexB,vertexA-vertexC,vertexA-incident.getOrigin()) / A;
	if(!(beta > 0.0 && gamma > 0.0 && beta+gamma < 1.0)) return Color(0.0,0.0,0.0);
	double alpha = 1.0 - (beta + gamma);
	// cout << alpha << " " << beta << " " << gamma << endl;
	return alpha*colors[0] + beta*colors[1] + gamma*colors[2];
}