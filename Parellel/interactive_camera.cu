#include <cmath>
#include "camera.h"
#include "interactive_camera.h"
#include "utilities.h"
#include "structures.h"

InteractiveCamera::InteractiveCamera() 
{   
	centerPosition = make_float3(0,0,0);
	yaw = 0;
	pitch = 0.3;
	radius = 10;
	apertureRadius = 0.04;

	resolution = make_float2(512,512);
	fov = make_float2(45, 45);
}

InteractiveCamera::~InteractiveCamera() {}

void InteractiveCamera::changeYaw(float m){
	yaw += m;
	fixYaw();
}

void InteractiveCamera::changePitch(float m){
	pitch += m;
	fixPitch();
}

void InteractiveCamera::changeRadius(float m){
	radius += radius * m; // Change proportional to current radius. Assuming radius isn't allowed to go to zero.
	fixRadius();
}

void InteractiveCamera::changeAltitude(float m){
	centerPosition.y += m;
	//fixCenterPosition();
}

void InteractiveCamera::changeApertureDiameter(float m){
	apertureRadius += (apertureRadius + 0.01) * m; // Change proportional to current apertureRadius.
	fixApertureRadius();
}

/*
void InteractiveCamera::changeFocalDistance(float m){
	focalDistance += m;
	fixFocalDistance();
}
*/

void InteractiveCamera::setResolution(float x, float y){
	resolution = make_float2(x,y);
	setFOVX(fov.x);
}

void InteractiveCamera::setFOVX(float fovx){
	fov.x = fovx;
	fov.y = ( atan( tan((fovx*M_PI / 180.0) * 0.5) * (resolution.y / resolution.x) ) * 2.0 ) * 180.0 / M_PI;
	//fov.y = (fov.x*resolution.y)/resolution.x; // TODO: Fix this! It's not correct! Need to use trig!
}

void InteractiveCamera::buildRenderCamera(Camera* renderCamera){
	float xDirection = sin(yaw) * cos(pitch);
	float yDirection = sin(pitch);
	float zDirection = cos(yaw) * cos(pitch);
	float3 directionToCamera = make_float3(xDirection, yDirection, zDirection);
	float3 viewDirection = directionToCamera * -1.0f;
	float3 eyePosition = centerPosition + directionToCamera * radius;

	renderCamera->setCameraVariables(eyePosition,viewDirection, make_float3(0, 1, 0), fov.y, resolution.x, resolution.y);

	//renderCamera->position = eyePosition; //make_float3(eyePosition[0], eyePosition[1], eyePosition[2]);
	//renderCamera->target = viewDirection; //make_float3(viewDirection[0], viewDirection[1], viewDirection[2]);
	//renderCamera->up = make_float3(0, 1, 0);
	//renderCamera->resolution = make_float2();
	//renderCamera->fov = make_float2(fov.x, fov.y);
	//renderCamera->apertureRadius = apertureRadius;
	//renderCamera->focalDistance = radius;
}

void InteractiveCamera::fixYaw() {
	yaw -= 2 * M_PI * floor(yaw / (2*M_PI)); // Normalize the yaw.
}

void InteractiveCamera::fixPitch() {
	float padding = 0.05;
	pitch = clamp(pitch, - (M_PI / 2) + padding, (M_PI / 2) - padding); // Limit the pitch.
}

void InteractiveCamera::fixRadius() {
	float minRadius = 0.2;
	float maxRadius = 100.0;
	radius = clamp(radius, minRadius, maxRadius);
}

void InteractiveCamera::fixApertureRadius() {
	float minApertureRadius = 0.0;
	float maxApertureRadius = 25.0;
	apertureRadius = clamp(apertureRadius, minApertureRadius, maxApertureRadius);
}

/*
void InteractiveCamera::fixFocalDistance() {
	float minRadius = 0.2;
	float maxRadius = 100.0;
	radius = BasicMath::clamp(radius, minRadius, maxRadius);
}
*/