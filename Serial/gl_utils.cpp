#include "gl_utils.h"

#ifdef __APPLE__
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#endif
#include <stdio.h>

int printOglError(const char *file, int line)
{
	GLenum glErr;
	int retCode = 0;
	glErr = glGetError();
	if(glErr != GL_NO_ERROR)
	{
		fprintf(stderr, "glError in file %s @ line %d: %s\n", file, line, gluErrorString(glErr));
		retCode = 1;
	}
	return retCode;
}