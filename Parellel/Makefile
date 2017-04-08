UNAME_S := $(shell uname)
EXTRA=
ifeq ($(UNAME_S), Darwin)
LDFLAGS = -Xlinker -framework,OpenGL -Xlinker -framework,GLUT
else
LDFLAGS += -L/usr/local/cuda/samples/common/lib/linux/x86_64
LDFLAGS += -lglut -lGL -lGLU
endif

NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-declarations -Wno-deprecated-gpu-targets" -dc --gpu-architecture=compute_35 --gpu-code=sm_35

all: main

main: main.o kernel.o structures.o utilities.o camera.o interactive_camera.o interactions.o
	$(NVCC) $^ -o $@ $(LDFLAGS) $(EXTRA) --gpu-architecture=compute_35 --gpu-code=sm_35

main.o: main.cu kernel.h interactions.h structures.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(EXTRA)

kernel.o: kernel.cu kernel.h structures.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(EXTRA)

structures.o: structures.cu structures.h utilities.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(EXTRA)

utilities.o: utilities.cu utilities.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(EXTRA)

camera.o: camera.cu camera.h structures.h utilities.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(EXTRA)

interactive_camera.o: interactive_camera.cu interactive_camera.h camera.h 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(EXTRA)

interactions.o: interactions.cu interactions.h 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(EXTRA) $(LDFLAGS)

clean:
	-rm -f main main.o kernel.o structures.o utilities.o
