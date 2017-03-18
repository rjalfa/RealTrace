# #OS X compilation flags
CPPFLAGS=-Wno-deprecated-declarations -I.
LDFLAGS= -framework GLUT -framework OpenGL -lstdc++ -O3 -lIL -lILU

#Linux compilation flags
CPPFLAGS=
LDFLAGS= -L/usr/X11R6/lib -L/usr/local/lib -lGL -lGLU -lm -lglut -lGLEW -lstdc++ -lIL -lILU

assignment05: 
	g++ -std=c++11 -o assignment05 lumina.cpp shader_utils.cpp utilities.cpp gl_utils.cpp camera.cpp vector3D.cpp world.cpp sphere.cpp triangle.cpp plane.cpp cylinder.cpp renderengine.cpp color.cpp ray.cpp material.cpp ${CPPFLAGS} ${LDFLAGS}

 clean:
	-rm -f assignment05
.PHONY: assignment05
