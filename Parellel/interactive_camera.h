#ifndef INTERACTIVE_CAMERA_H
#define INTERACTIVE_CAMERA_H

class Camera;

class InteractiveCamera
{
private:

	float3 centerPosition;
	float yaw;
	float pitch;
	float radius;
	float apertureRadius;

	void fixYaw();
	void fixPitch();
	void fixRadius();
	void fixApertureRadius();

public:
	InteractiveCamera();
	virtual ~InteractiveCamera();
   	void changeYaw(float m);
	void changePitch(float m);
	void changeRadius(float m);
	void changeAltitude(float m);
	void changeApertureDiameter(float m);
	void setResolution(float x, float y);
	void setFOVX(float fovx);

	void buildRenderCamera(Camera* renderCamera);

	float2 resolution;
	float2 fov;
};

#endif // INTERACTIVE_CAMERA_H