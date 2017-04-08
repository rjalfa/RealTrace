#include "interactions.h"

InteractiveCamera* interaction;
Camera* h_camera;
////////////////////////////////////////////////////////////////////////////////
// Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
int lastX = 0, lastY = 0;
int theButtonState = 0;
int theModifierState = 0;

void motion(int x, int y)
{
	int deltaX = lastX - x;
	int deltaY = lastY - y;

	if (deltaX != 0 || deltaY != 0) {

		//bool moveLeftRight = abs(deltaX) > abs(deltaY);
		//bool moveUpDown = !moveLeftRight;

		if (theButtonState == GLUT_LEFT_BUTTON)  // Rotate
		{
			interaction->changeYaw(deltaX * 0.01);
			interaction->changePitch(-deltaY * 0.01);
		}
		else if (theButtonState == GLUT_MIDDLE_BUTTON) // Zoom
		{
			interaction->changeAltitude(-deltaY * 0.01);
		}    

		if (theButtonState == GLUT_RIGHT_BUTTON) // camera move
		{
			interaction->changeRadius(-deltaY * 0.01);

			if (theModifierState & GLUT_ACTIVE_ALT) // Pan
			{

			}   
		}
 
		lastX = x;
		lastY = y;
		glutPostRedisplay(); // Is this necessary?

	}

}

void mouse(int button, int state, int x, int y)
{
	theButtonState = button;
	lastX = x;
	lastY = y;

	motion(x, y);
}

void keyboard(unsigned char key, int x, int y) {
   if (key == 27)  exit(0);
   glutPostRedisplay();
 }