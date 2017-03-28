//http://www.informit.com/articles/article.aspx?p=2455391&seqNum=3
#include "kernel.h"
#include "structures.h"
#include "camera.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
//#include <GL/glew.h>
#include <GL/freeglut.h>
#define DEFAULT_COLOR make_float3(0.8,0.0,0.4)
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "helper_cuda.h"
#include "helper_cuda_gl.h"
#include "interactions.h"
#include "helper_gl.h"
#define SCALING_FACTOR 2

using namespace std;
 // texture and pixel objects
 GLuint pbo = 0;     // OpenGL pixel buffer object
 GLuint tex = 0;     // OpenGL texture object
 struct cudaGraphicsResource *cuda_pbo_resource;

int screen_width = W;
int screen_height = H;

Ray* d_rays;
vector<Ray> h_rays;
Triangle* d_triangles;
std::vector<Triangle> h_triangles;
LightSource* d_light, *h_light;
int num_triangles;
Camera *camera;

void render() {
   uchar4 *d_out = 0;
   cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
   cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource);
   kernelLauncher(d_out, W, H, d_rays, d_triangles, num_triangles, d_light);
   cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
   // update contents of the title bar
   char title[64];
   sprintf(title, "RealTrace [TM] | Real-Time Raytracer | CUDA");
   glutSetWindowTitle(title);
 }
void drawTexture() {
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
  glTexCoord2f(0.0f, 1.0f); glVertex2f(0, H);
  glTexCoord2f(1.0f, 1.0f); glVertex2f(W, H);
  glTexCoord2f(1.0f, 0.0f); glVertex2f(W, 0);
  glEnd();
  glDisable(GL_TEXTURE_2D);
}

void display() {
  render();
  drawTexture();
  glutSwapBuffers();
}

void initGLUT(int *argc, char **argv) {
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(W, H);
  glutCreateWindow(TITLE_STRING);
}

void initPixelBuffer() {
  //cerr << "Hello" << endl;
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, 4*W*H*sizeof(GLubyte), 0,
               GL_STREAM_DRAW);
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
                               cudaGraphicsMapFlagsWriteDiscard);
}

void readData(string file_name, string texture_file_name = "", string occlusion_map_file_name = "")
{
  ifstream is(file_name.c_str());
  if(!is.is_open()) {
    cerr << "could not open file" << endl;
    exit(0);
  }

  vector < float3 > vertices;
  vector < float3 > normal_vertices;
  vector < pair<double, double> > texture_vertices;

  string c;
  double v[3];

  while(is >> c) {
    if(c == "f") {
      vector < int > idx[3];
      string data, token;
      for(int i = 0; i < 3; i++) {
        is >> data;
        stringstream ss(data);
        while(getline(ss, token, '/')) {
          idx[i].push_back(atoi(token.c_str()));
        }
      }
      Triangle t;
      t.vertexA = vertices[idx[0][0]-1];
      t.vertexB = vertices[idx[1][0]-1];
      t.vertexC = vertices[idx[2][0]-1];
      t.color = DEFAULT_COLOR;
      h_triangles.push_back(t);
      // cerr << "rendered\n";
    } else if(c == "v") {
      is >> v[0] >> v[1] >> v[2];
      vertices.push_back(make_float3(v[0]*SCALING_FACTOR, v[1]*SCALING_FACTOR, v[2]*SCALING_FACTOR));
    } else if(c == "vn") {
      is >> v[0] >> v[1] >> v[2];
      normal_vertices.push_back(make_float3(v[0], v[1], v[2]));
    } else if(c == "vt") {
      is >> v[0] >> v[1];
      texture_vertices.push_back(make_pair((double)v[0],(double)v[1]));
    } else if(c[0] == '#') {
      getline(is, c);
    } else {
      getline(is, c);
    }
  }
  is.close();

  float3 camera_position = make_float3(60, 60, 0);
  float3 camera_target = make_float3(0, 0, 0); //Looking down -Z axis
  float3 camera_up = make_float3(0, 1, 0);
  float camera_fovy =  45;
  Camera* camera = new Camera(camera_position, camera_target, camera_up, camera_fovy, screen_width, screen_height);
  
  //Create Ray array
  for(int i = 0; i < screen_width; i ++) for(int j = 0; j < screen_height; j ++)
  {
    float3 ray_dir = camera->get_ray_direction(i, j);
    Ray ray;
    ray.origin = camera->get_position();
    ray.direction = ray_dir;
    h_rays.push_back(ray);
  }

  //Create Light Source
  h_light = new LightSource;
  h_light->position = make_float3(-10,-10,0);
  h_light->color = make_float3(1,1,1);

  //Memcpy to GPU

  checkCudaErrors(cudaMalloc((void**)&d_rays, sizeof(Ray)*h_rays.size()));
  checkCudaErrors(cudaMalloc((void**)&d_light, sizeof(LightSource)));
  checkCudaErrors(cudaMalloc((void**)&d_triangles, sizeof(Triangle)*h_triangles.size()));
  // cudaMalloc((void**)&d_rays, sizeof(Ray)*rays.size()));
  long long mem = sizeof(Ray)*h_rays.size() + sizeof(LightSource) + sizeof(Triangle)*h_triangles.size();
  checkCudaErrors(cudaMemcpy(d_rays,&h_rays[0],sizeof(Ray)*h_rays.size(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_light,h_light,sizeof(LightSource), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_triangles,&h_triangles[0], sizeof(Triangle)*h_triangles.size(), cudaMemcpyHostToDevice));
  cerr << "[INFO] Memory to be transferred to GPU: " << mem << " B" << endl;
  num_triangles = h_triangles.size();  
}

void exitfunc() {
  if (pbo) {
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
  }
  delete h_light;
  delete camera;
  cudaFree(d_light);
  cudaFree(d_triangles);
  cudaFree(d_rays);
}

int main(int argc, char** argv) {
  // printInstructions();
  //glewInit();
string filename = "bob_tri.obj";
if(argc > 1) filename = string(argv[1]);  
readData(filename);
  initGLUT(&argc, argv);
  gluOrtho2D(0, W, H, 0);
  glutKeyboardFunc(keyboard);
  glutSpecialFunc(handleSpecialKeypress);
  glutPassiveMotionFunc(mouseMove);
  glutMotionFunc(mouseDrag);
  glutDisplayFunc(display);
  initPixelBuffer();
  glutMainLoop();
  atexit(exitfunc);
  return 0;
}
