UNAME_S := $(shell uname)
ifeq ($(UNAME_S), Darwin)
LDFLAGS = -Xlinker -framework,OpenGL -Xlinker -framework,GLUT
else
LDFLAGS += -L/usr/local/cuda/samples/common/lib/linux/x86_64
LDFLAGS += -lglut -lGL -lGLU
endif

NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-declarations -Wno-deprecated-gpu-targets" -dc

all: main

main: main.o kernel.o structures.o utilities.o camera.o
	$(NVCC) $^ -o $@ $(LDFLAGS)

main.o: main.cu kernel.h interactions.h structures.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

kernel.o: kernel.cu kernel.h structures.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

structures.o: structures.cu structures.h utilities.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

utilities.o: utilities.cu utilities.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

camera.o: camera.cu camera.h structures.h utilities.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	-rm -f main main.o kernel.o structures.o utilities.o
