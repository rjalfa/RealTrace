#ifndef KERNEL_H
#define KERNEL_H
 
struct Ray;
class Triangle;
class uchar4;
struct LightSource;

void kernelLauncher(uchar4 *d_out, int w, int h, Ray* rays, Triangle* triangles, int num_triangles, LightSource* l); 

#endif