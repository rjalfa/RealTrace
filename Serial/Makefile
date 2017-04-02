EXTRA=
# #OS X compilation flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CXXFLAGS=-Wno-deprecated-declarations -I.
  LDFLAGS= -framework GLUT -framework OpenGL -lstdc++ -O3 -lIL -lILU
endif
ifeq ($(UNAME_S),Linux)
  CXXFLAGS= -std=c++14 -I.
  ifdef EXTRA 
    LDFLAGS= -L/usr/X11R6/lib -L/usr/local/lib -lGL -lGLU -lm -lglut -lstdc++ -DCUDA_SERVER
  else
    LDFLAGS= -L/usr/X11R6/lib -L/usr/local/lib -lGL -lGLU -lm -lglut -lGLEW -lstdc++ -lIL -lILU
  endif

endif
OBJ = color.o utilities.o gl_utils.o uniform-grid.o camera.o vector3D.o world.o sphere.o triangle.o plane.o cylinder.o renderengine.o ray.o material.o
EXEC = lumina

all: $(EXEC)

$(OBJ): %.o : %.cpp %.h
	$(CXX) -c $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(EXEC): % : %.cpp $(OBJ)
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)

clean:
	-rm -f $(OBJ) $(EXEC)
.PHONY: main
