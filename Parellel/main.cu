//http://www.informit.com/articles/article.aspx?p=2455391&seqNum=3
#include "kernel.h"
#include "structures.h"
#include "camera.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <sys/time.h>
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
#endif
#define DEFAULT_COLOR make_float3(0.8,0.7,0.0)
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "interactive_camera.h"
#include "interactions.h"
#ifndef _NO_OPENGL
#include <cuda_gl_interop.h>
#include "helper_cuda_gl.h"
#include "helper_gl.h"
#endif
#ifndef _NO_OPENGL
#define OPENGL(X) X
#else
#define OPENGL(X)
#endif

#define SCALING_FACTOR 2

using namespace std;
// texture and pixel objects
OPENGL(
	GLuint pbo = 0;     // OpenGL pixel buffer object
	GLuint tex = 0;     // OpenGL texture object
	struct cudaGraphicsResource *cuda_pbo_resource;
);
int num_max = 10000000;
int screen_width = 512;
int screen_height = 512;

Triangle* d_triangles;
std::vector<Triangle> h_triangles;
LightSource* d_light, *h_light;
int num_triangles;
Camera *d_camera = NULL;

long frames = 0;
uchar4 *d_out = 0;

void render() {
	OPENGL(
		cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
		cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource););
		kernelLauncher(d_out, screen_width, screen_height, d_camera, d_triangles, num_triangles, d_light
	);
	OPENGL(
		cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
		//cudaDeviceSynchronize();
		// update contents of the title bar
		char title[64];
		sprintf(title, "RealTrace [TM] | Real-Time Raytracer | CUDA");
		glutSetWindowTitle(title);
	);
	frames ++ ;
}

OPENGL(
	void show_fps(int a)
	{
		cout << "\rFPS: " << frames * (1000.0 / a);
		frames = 0;
	}
);

OPENGL(
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
	int theta = 0;
	void display() {
		//fprintf(stderr,"Reder Func\n");
		//theta = (theta + 10)%360;
		//float3 camera_position = make_float3(60*cos((theta * 3.141592)/180.0), 0, 60*sin((theta * 3.141592)/180.0));
		//printf("Camera Add: %p\n", h_camera);
		interaction->buildRenderCamera(h_camera);
		//h_camera->setCameraPosition(camera_position);
		checkCudaErrors(cudaMemcpy(d_camera, h_camera, sizeof(Camera), cudaMemcpyHostToDevice));
		render();
		drawTexture();
		glutSwapBuffers();
		glutPostRedisplay();
	}

	void initGLUT(int *argc, char **argv) {
		glutInit(argc, argv);
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
		glutInitWindowSize(W, H);
		glutCreateWindow(TITLE_STRING);
	}

	void initPixelBuffer() {
		//cerr << "Hello" << endl;
	#ifndef _NO_OPENGL
		glGenBuffers(1, &pbo);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * W * H * sizeof(GLubyte), 0,
					 GL_STREAM_DRAW);
		glGenTextures(1, &tex);
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
									 cudaGraphicsMapFlagsWriteDiscard);
	#else
		checkCudaErrors(cudaMalloc((void**)&d_out, uchar4 * W * H));
	#endif
	}
)

void readData(string file_name, string texture_file_name = "", string occlusion_map_file_name = "")
{
	ifstream is(file_name.c_str());
	if (!is.is_open()) {
		cerr << "could not open file" << endl;
		exit(0);
	}

	vector < float3 > vertices;
	vector < float3 > normal_vertices;
	vector < pair<double, double> > texture_vertices;

	string c;
	double v[3];

	while (is >> c) {
		if (c == "f") {
			vector < int > idx[3];
			string data, token;
			for (int i = 0; i < 3; i++) {
				is >> data;
				stringstream ss(data);
				while (getline(ss, token, '/')) {
					if (token != "")
						idx[i].push_back(atoi(token.c_str()));
				}
			}
			Triangle t;
			t.vertexA = vertices[idx[0][0] - 1] + make_float3(-5, 0, 0);
			t.vertexB = vertices[idx[1][0] - 1] + make_float3(-5, 0, 0);
			t.vertexC = vertices[idx[2][0] - 1] + make_float3(-5, 0, 0);
			t.color = DEFAULT_COLOR;
			t.type_of_material = REFRACTIVE;
			Triangle t1;
			t1.vertexA = vertices[idx[0][0] - 1] + make_float3(5, 0, 0);
			t1.vertexB = vertices[idx[1][0] - 1] + make_float3(5, 0, 0);
			t1.vertexC = vertices[idx[2][0] - 1] + make_float3(5, 0, 0);
			t1.color = make_float3(0,0.1,0.6);
			t1.type_of_material = REFRACTIVE;

			Triangle t2;
			t2.vertexA = vertices[idx[0][0] - 1] + make_float3(0, 5, 0);
			t2.vertexB = vertices[idx[1][0] - 1] + make_float3(0, 5, 0);
			t2.vertexC = vertices[idx[2][0] - 1] + make_float3(0, 5, 0);
			t2.color = make_float3(0.4,0.1,0.6);
			t2.type_of_material = REFRACTIVE;

			if (h_triangles.size() < static_cast<unsigned int>(num_max)) h_triangles.push_back(t);
			if (h_triangles.size() < static_cast<unsigned int>(num_max)) h_triangles.push_back(t1);
			if (h_triangles.size() < static_cast<unsigned int>(num_max)) h_triangles.push_back(t2);
			// cerr << "rendered\n";
		} else if (c == "v") {
			is >> v[0] >> v[1] >> v[2];
			vertices.push_back(make_float3(v[0]*SCALING_FACTOR, v[1]*SCALING_FACTOR, v[2]*SCALING_FACTOR));
		} else if (c == "vn") {
			is >> v[0] >> v[1] >> v[2];
			normal_vertices.push_back(make_float3(v[0], v[1], v[2]));
		} else if (c == "vt") {
			is >> v[0] >> v[1];
			texture_vertices.push_back(make_pair((double)v[0], (double)v[1]));
		} else if (c[0] == '#') {
			getline(is, c);
		} else {
			getline(is, c);
		}
	}
	is.close();

	//Add Floor
	Triangle f1;
	f1.vertexA = make_float3(20, 20, 0);
	f1.vertexB = make_float3(20, -20, 0);
	f1.vertexC = make_float3(-20, 20, 0);
	f1.color = make_float3(0.5, 0.5, 1.0);
	f1.type_of_material = REFLECTIVE;
	Triangle f2;
	f2.vertexA = make_float3(20, -20, 0);
	f2.vertexB = make_float3(-20, 20, 0);
	f2.vertexC = make_float3(-20, -20, 0);
	f2.color = make_float3(0.5, 0.5, 1.0);
	f2.type_of_material = REFLECTIVE;
	h_triangles.push_back(f1); h_triangles.push_back(f2);

	float3 camera_position = make_float3(60, 0, 60);
	float3 camera_target = make_float3(0, 0, 0); //Looking down -Z axis
	float3 camera_up = make_float3(0, -1, 0);
	float camera_fovy =  45;
	h_camera = new Camera(camera_position, camera_target, camera_up, camera_fovy, screen_width, screen_height);
	interaction = new InteractiveCamera();
	interaction->buildRenderCamera(h_camera);
	cudaHostRegister(h_camera, sizeof(Camera), 0);
	//Create Light Source
	h_light = new LightSource;
	h_light->position = make_float3(-10, -10, 0);
	h_light->color = make_float3(1, 1, 1);

	//Memcpy to GPU

	checkCudaErrors(cudaMalloc((void**)&d_light, sizeof(LightSource)));
	checkCudaErrors(cudaMalloc((void**)&d_triangles, sizeof(Triangle)*h_triangles.size()));
	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera)));
	// cudaMalloc((void**)&d_rays, sizeof(Ray)*rays.size()));
	long long mem = sizeof(Camera) + sizeof(LightSource) + sizeof(Triangle) * h_triangles.size();
	checkCudaErrors(cudaMemcpy(d_camera, h_camera, sizeof(Camera), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_light, h_light, sizeof(LightSource), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_triangles, &(h_triangles[0]), sizeof(Triangle)*h_triangles.size(), cudaMemcpyHostToDevice));
	//cudaDeviceSynchronize();
	cerr << "[INFO] Memory to be transferred to GPU: " << mem << " B" << endl;
	num_triangles = h_triangles.size();
	cerr << "[INFO] readData Complete" << endl;
	buildGrid(screen_width, screen_height, d_triangles, num_triangles);
	create_space_for_kernels(screen_width, screen_height);
}

void exitfunc() {
	OPENGL(
	if (pbo) {
	cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}
	cout << endl;
	);

#ifdef _NO_OPENGL
	checkCudaErrors(cudaFree(d_out));
#endif
	delete h_light;
	cudaHostUnregister(h_camera);
	delete h_camera;
	delete interaction;
	// free_space_for_kernels();
	checkCudaErrors(cudaFree(d_light));
	checkCudaErrors(cudaFree(d_triangles));
	checkCudaErrors(cudaFree(d_camera));
}

int main(int argc, char** argv) {
	// printInstructions();
	//glewInit();
	string filename = "bob_tri.obj";
	if (argc > 1) filename = string(argv[1]);
	if (argc > 2) num_max = atoi(argv[2]);
	readData(filename);
	OPENGL(
		initGLUT(&argc, argv);
		gluOrtho2D(0, W, H, 0);
		glutKeyboardFunc(keyboard);
		// glutSpecialFunc(handleSpecialKeypress);
		glutMotionFunc(motion);
		glutMouseFunc(mouse);
		glutTimerFunc(1000, show_fps, 1000);
		glutDisplayFunc(display);
		initPixelBuffer();
		glutMainLoop();
	);
#ifdef _NO_OPENGL
	render();
	cudaDeviceSynchronize();
#endif
	atexit(exitfunc);
	return 0;
}
