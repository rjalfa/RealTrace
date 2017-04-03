#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
 
struct Ray;
class Triangle;
class uchar4;
struct LightSource;
class Camera;
void kernelLauncher(uchar4 *d_out, int w, int h, Camera* camera, Triangle* triangles, int num_triangles, LightSource* l); 
void buildGrid(int w, int h, Triangle * triangles, int num_triangles);

#endif
