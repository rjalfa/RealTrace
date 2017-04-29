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
#define DEFAULT_COLOR make_float3(0.8,0.7,0.0)
#include <cuda_runtime.h>
#include "helper_cuda.h"
#define SCALING_FACTOR 15
#define W 512
#define H 512
using namespace std;

int num_max = 10000000;
int screen_width = W;
int screen_height = H;
Camera * h_camera;
Triangle* d_triangles;
std::vector<Triangle> h_triangles;
LightSource* d_light, *h_light;
int num_triangles;
Camera *d_camera = NULL;

uchar4 *d_out = 0;

void loadModel(string file_name, float scaling_factor = SCALING_FACTOR, float3 translation = make_float3(0,0,0), int type_of_material = DIFFUSE, float3 color = DEFAULT_COLOR)
{
	ifstream is(file_name.c_str());
	if (!is.is_open()) {
		cerr << "could not open file" << endl;
		return;
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
			t.vertexA = vertices[idx[0][0] - 1] + translation;
			t.vertexB = vertices[idx[1][0] - 1] + translation;
			t.vertexC = vertices[idx[2][0] - 1] + translation;
			t.color = color;
			t.type_of_material = type_of_material;

			if (h_triangles.size() < static_cast<unsigned int>(num_max)) h_triangles.push_back(t);
		} else if (c == "v") {
			is >> v[0] >> v[1] >> v[2];
			vertices.push_back(make_float3(v[0]*scaling_factor, v[1]*scaling_factor, v[2]*scaling_factor));
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
}

void setupData(string file_name, string texture_file_name = "", string occlusion_map_file_name = "")
{
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
	float3 camera_up = make_float3(0, 1, 0);
	float camera_fovy =  45;
	h_camera = new Camera(camera_position, camera_target, camera_up, camera_fovy, screen_width, screen_height);
	cudaHostRegister(h_camera, sizeof(Camera), 0);
	//Create Light Source
	h_light = new LightSource;
	h_light->position = make_float3(-10, -10, 0);
	h_light->color = make_float3(1, 1, 1);

	//Memcpy to GPU
	checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(uchar4) * W * H));
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

void export_frame(string out_file_name = "out.ppm")
{

	uchar4* h_out = new uchar4[W*H];
	checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(uchar4) * W * H, cudaMemcpyDeviceToHost));
	std::ofstream ofs; 
    ofs.open("out.ppm"); 
    ofs << "P6\n" << W << " " << H << "\n255\n"; 
    for(uint32_t j = 0; j < W; ++j) 
    	for (uint32_t i = 0; i < H; ++i) 
    { 
        char r = (char)h_out[i*W + j].x; 
        char g = (char)h_out[i*W + j].y; 
        char b = (char)h_out[i*W + j].z; 
        ofs << r << g << b; 
    } 
    ofs.close();
    delete[] h_out;
}

void exitfunc() {
	checkCudaErrors(cudaFree(d_out));
	delete h_light;
	cudaHostUnregister(h_camera);
	delete h_camera;
	free_space_for_kernels();
	checkCudaErrors(cudaFree(d_light));
	checkCudaErrors(cudaFree(d_triangles));
	checkCudaErrors(cudaFree(d_camera));
}

int main(int argc, char** argv) {
	// printInstructions();
	//glewInit();
	string out_file_name = "out.ppm";
	string filename = "bob_tri.obj";
	if (argc > 1) filename = string(argv[1]);
	if(argc > 2) out_file_name = string(argv[2]);
	if (argc > 3) num_max = atoi(argv[3]);
	//Load Model
	loadModel(filename, SCALING_FACTOR, make_float3(0,0,2), REFRACTIVE);
	//Setup GPU Memory
	setupData(filename);
	//Launch Kernel
	kernelLauncher(d_out, screen_width, screen_height, d_camera, d_triangles, num_triangles, d_light);
	//Export Image
	export_frame(out_file_name);
	//Cleanup
	exitfunc();
	return 0;
}
