#ifndef _GL_UTILS_H_
#define _GL_UTILS_H_

#define M_PI 3.14159265f

#define printOpenGLError() printOglError(__FILE__, __LINE__)
#define degreeToRadians(X) ((X)*M_PI/180.0f)

int printOglError(const char *file, int line);
#endif
