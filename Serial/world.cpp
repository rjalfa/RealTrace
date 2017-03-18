#include "world.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define EULER_CONSTANT 2.718282
using namespace std;

float World::firstIntersection(Ray& ray)
{
	for(int i=0; i<objectList.size(); i++)
	{
		objectList[i]->intersect(ray);
	}
	return ray.getParameter();
}

bool refract(const Vector3D& I,const Vector3D& N, const double eta, Vector3D& T)
{
	double k = 1.0 - eta * eta * (1.0 - dotProduct(N, I) * dotProduct(N, I));
	if(k < 0) return false;
	T = eta * I - (eta * dotProduct(N, I) + sqrt(k)) * N;
	return true;
}

Vector3D reflect(const Vector3D& I, const Vector3D& N)
{
	return I-2.0*dotProduct(N, I)*N;
}

Color World::shade_ray(Ray ray)
{
	if(ray.getLevel() > RECURSION_DEPTH) return background;
	firstIntersection(ray);
	if(ray.didHit())
	{
		Color shadowColor(0.0,0.0,0.0);
		bool isShadow = false;
		//Run Shadow Test
		const Object* intersectedObject = ray.intersected();
		for(LightSource* ls : this->lightSourceList) {
			Ray shadowRay(ray.getPosition()+0.01*(ls->getPosition()-ray.getPosition()),ls->getPosition()-ray.getPosition());
			firstIntersection(shadowRay);
			if(shadowRay.didHit()) {
				isShadow = true;
				shadowColor = ambient*(intersectedObject->getMaterial()->shade(ray))*(intersectedObject->getMaterial())->ka;
			}
		}
		//..Compute Shade factor due to light
		Color lightColor(0.0,0.0,0.0);
		for(LightSource* ls : this->lightSourceList) {
			lightColor = lightColor + get_light_shade(ray.getPosition(),intersectedObject->getNormalAtPosition(ray.getPosition()),*ls,intersectedObject->getMaterial(),ray.getDirection());
		}
		lightColor = lightColor + ambient*(intersectedObject->getMaterial()->shade(ray))*(intersectedObject->getMaterial())->ka;
		//if(shadowEffect) lightColor = lightColor*intersectedObject->getMaterial()->ka;
		
		Color finalColor = lightColor;
		if(isShadow) finalColor = finalColor*(1e-4) + shadowColor*(1 - 1e-4);

		//Reflection
		auto N = intersectedObject->getNormalAtPosition(ray.getPosition());
		auto I = ray.getDirection();
		N.normalize();
		I.normalize();

		double eta = intersectedObject->getMaterial()->eta;
		Vector3D T(0.0,0.0,0.0);
		double t = ray.getParameter();
		double c = 0;
		Vector3D k(1.0,1.0,1.0);
		int level = ray.getLevel();
		if(intersectedObject->getMaterial()->kr > 0 && intersectedObject->getMaterial()->kt > 0)
		{
			//Dielectrics 
			auto R = reflect(I,N);
			if(dotProduct(ray.getDirection(),N) < 0)
			{
				refract(I,N,eta,T);
				c = -dotProduct(I,N);
			}
			else
			{
				k = Vector3D(pow(EULER_CONSTANT,-1.0*0.27*t),pow(EULER_CONSTANT,-1.0*0.45*t),pow(EULER_CONSTANT,-1.0*0.55*t));
				if(refract(I,-1.0*N,1/eta,T)) c = dotProduct(T,N);
				else {
					Ray temp = Ray(ray.getPosition()+ 1e-4 * R,R,level+1);
					return k*shade_ray(temp);
				}
			}
			double _R0 = ((eta-1)*(eta-1))/((eta+1)*(eta+1));
			double _R = _R0 + (1-_R0)*pow(1-c,5);
			Ray temp1 = Ray(ray.getPosition()+ 1e-4 * R,R,level+1);
			Ray temp2 = Ray(ray.getPosition()+ 1e-4 * T,T,level+1);
			return k*(_R * shade_ray(temp1) + (1-_R)*shade_ray(temp2));
		}
		else if(intersectedObject->getMaterial()->kr > 0)
		{
			auto R = reflect(I,N);
			Ray reflectedRay(ray.getPosition()+ 1e-4 * R,R);
			finalColor = finalColor + (intersectedObject->getMaterial()->kr)*shade_ray(reflectedRay);
		}
		return finalColor;
	}
	return background;
}

//Returns Sum of diffuse and specular reflections at position due to light source.
Color World::get_light_shade(const Vector3D& position, const Vector3D& normal, const LightSource& lightSource,const Material* mat, const Vector3D viewVector)
{
	using namespace glm;
	vec3 vNormal = vec3(normal.X(),normal.Y(),normal.Z());
	vec3 vVertex = vec3(position.X(),position.Y(),position.Z());
	vec3 vLightPosition = vec3(lightSource.getPosition().X(),lightSource.getPosition().Y(),lightSource.getPosition().Z());
	vec3 vViewVector = vec3(viewVector.X(),viewVector.Y(),viewVector.Z());
	vec3 n = normalize(vNormal);
	vec3 r = normalize(reflect(-normalize(vLightPosition-vVertex),n));
	float dist = distance(vVertex,vLightPosition);
	float fatt = 1.0 / (1.0 + 0.05*dist);
	float diffuse = std::max(dot(n,normalize(vLightPosition)),0.0f);
	float specular = std::max(std::pow(dot(normalize(vViewVector),r),mat->n),0.0);
	return (mat->kd)*diffuse*(lightSource.getIntensity())*(mat->shade(Ray(position,viewVector))) + (mat->ks)*specular*(lightSource.getIntensity());
}

