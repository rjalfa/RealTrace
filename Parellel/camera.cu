#include "camera.h"
#include "structures.h"
#include <math.h>

__host__ float3 normalizeh(float3 a)
{
	float len = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
	return make_float3(a.x / len, a.y / len, a.z/len);
}

Camera::Camera(const float3& _pos, const float3& _target, const float3& _up, float _fovy, int _width, int _height) :
	position(_pos), target(_target), up(_up), fovy(_fovy), width(_width), height(_height)
{
	up = normalizeh(up);

	line_of_sight = target - position;

	//Calculate the camera basis vectors
	//Camera looks down the -w axis
	w = normalizeh(-line_of_sight);
	u = normalizeh(crossProduct(up, w));
	v = normalizeh(crossProduct(w, u));

	focalHeight = 1.0; //Let's keep this fixed to 1.0
	aspect = float(width) / float(height);
	focalWidth = focalHeight * aspect; //Height * Aspect ratio
	focalDistance = focalHeight / (2.0 * tan(fovy * M_PI / (180.0 * 2.0))); //More the fovy, close is focal plane
}

Camera::~Camera()
{

}

//Get direction of viewing ray from pixel coordinates (i, j)
__device__ const float3 Camera::get_ray_direction(const int i, const int j) const
{
	float3 dir = make_float3(0.0, 0.0, 0.0);
	dir = dir + (-w * focalDistance);
	float xw = aspect * (i - width / 2.0 + 0.5) / width;
	float yw = (j - height / 2.0 + 0.5) / height;
	dir = dir + u * xw;
	dir = dir + v * yw;

	return normalize(dir);
}

void Camera::setCameraVariables(const float3& _pos, const float3& _target, const float3& _up, float _fovy, int _width, int _height)
{

	position = _pos;
	up = _up;
	target = _target;
	fovy = _fovy;
	width = _width;
	height = _height;

	up = normalizeh(up);

	line_of_sight = target - position;

	//Calculate the camera basis vectors
	//Camera looks down the -w axis
	w = normalizeh(-line_of_sight);
	u = normalizeh(crossProduct(up, w));
	v = normalizeh(crossProduct(w, u));

	focalHeight = 1.0; //Let's keep this fixed to 1.0
	aspect = float(width) / float(height);
	focalWidth = focalHeight * aspect; //Height * Aspect ratio
	focalDistance = focalHeight / (2.0 * tan(fovy * M_PI / (180.0 * 2.0)));
}
