#include "camera.h"
#include "structures.h"
#include <math.h>

Camera::Camera(const float3& _pos, const float3& _target, const float3& _up, float _fovy, int _width, int _height) : 
position(_pos), target(_target), up(_up), fovy(_fovy), width(_width), height(_height)
{
	up = normalize(up);

	line_of_sight = target - position;

	//Calculate the camera basis vectors
	//Camera looks down the -w axis
	w = normalize(-line_of_sight);
	u = normalize(crossProduct(up, w));
	v = normalize(crossProduct(w, u));

	focalHeight = 1.0; //Let's keep this fixed to 1.0
	aspect = float(width)/float(height);
	focalWidth = focalHeight * aspect; //Height * Aspect ratio
	focalDistance = focalHeight/(2.0 * tan(fovy * M_PI/(180.0 * 2.0))); //More the fovy, close is focal plane
}

Camera::~Camera()
{

}

//Get direction of viewing ray from pixel coordinates (i, j)
const float3 Camera::get_ray_direction(const int i, const int j) const
{
	float3 dir = make_float3(0.0, 0.0, 0.0);
	dir = dir + (-w * focalDistance);
	float xw = aspect*(i - width/2.0 + 0.5)/width;
	float yw = (j - height/2.0 + 0.5)/height;
	dir = dir + u * xw;
	dir = dir + v * yw;

	return normalize(dir);
}
