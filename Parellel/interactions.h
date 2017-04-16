#ifndef INTERACTIONS_H
 #define INTERACTIONS_H
 #define W 512
 #define H 512
 #define DELTA_P 0.1f
 #define TITLE_STRING "Stability"

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
//#include <GL/glew.h>
#include <GL/freeglut.h>
#endif
#include "interactive_camera.h"
#include "camera.h"

 extern InteractiveCamera* interaction;
 extern Camera* h_camera;

 void keyboard(unsigned char key, int x, int y);
// no mouse interactions implemented for this app
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

 #endif
