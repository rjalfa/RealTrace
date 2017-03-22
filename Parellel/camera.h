#ifndef _CAMERA_H_
#define _CAMERA_H_

class Camera
{
private:
	float3 position;
	float3 target; //Look-at point
	float3 up;

	float3 line_of_sight;
	float3 u, v, w; //Camera basis vectors

	unsigned char *bitmap;
	int width, height;
	float fovy;// expressed in degrees: FOV-Y; angular extent of the height of the image plane
	float focalDistance; //Distance from camera center to the image plane
	float focalWidth, focalHeight;//width and height of focal plane
	float aspect;

public:
	Camera(const float3& _pos, const float3& _target, const float3& _up, float fovy, int w, int h);
	~Camera();
	const float3 get_ray_direction(const int i, const int j) const;
	const float3& get_position() const { return position; }
	int getWidth() {return width;}
	int getHeight(){return height;}

};
#endif
