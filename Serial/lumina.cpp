#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/freeglut.h>
#endif

#include <IL/il.h>
#include <IL/ilu.h>

#include "shader_utils.h"
#include "gl_utils.h"

#include "camera.h"
#include "renderengine.h"
#include "world.h"
#include "material.h"
#include "object.h"
#include "sphere.h"
#include "triangle.h"
#include "cylinder.h"
#include "plane.h"
#include "lightsource.h"
#include "pointlightsource.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>

#define SCALING_FACTOR 5

//Globals
GLuint program;
GLint attribute_coord2d;
int screen_width = 800, screen_height = 600; //Both even numbers
float quadVertices[] = {-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1}; //2D screen space coordinates 
GLuint texImage;
GLint uniform_texImage;

Camera *camera;
RenderEngine *engine;

void init_material_from_obj(Material * m) {
	m->color = Color(0.8, 0.1, 0.0);
	m->ka = 0.2;
	m->kd = 0.9;
	m->ks = 0.4;
	m->kr = 0.0;
	m->kt = 0.0;
	m->eta = 1.0;
	m->n = 128;
}

void get_value_by_coordinate()
{

}

void load_image_from_obj(World * world, string file_name, string texture_file_name = "", string occlusion_map_file_name = "") {
	ifstream is(file_name);
	if(!is.is_open()) {
		cerr << "could not open file" << endl;
		exit(0);
	}

	bool has_texture_map = false;

	ILuint imageID = -1;
	
	if(texture_file_name != "") {
		//Open and Load Texture file via devIL
    	ilGenImages(1, &imageID);
		ilBindImage(imageID);
		if(!ilLoadImage(texture_file_name.c_str()))
		{
			cerr << "Internal Texture Load Error" << endl;
			exit(1);
		}

		ILenum devilError = ilGetError();
		if (devilError != IL_NO_ERROR) {
    		cerr << "Devil Error: " << iluErrorString(devilError) << endl;
    		exit(1);
		}
		
		
		exit(0);
		has_texture_map = true;
	}

	vector < Vector3D > vertices;
	vector < Vector3D > normal_vertices;
	vector < pair<double, double> > texture_vertices;

	string c;
	double v[3];
	vector < int > idx[3];
	// Material *m1 = new Material(world);

	while(is >> c) {
		if(c == "f") {
			string data, token;
			for(int i = 0; i < 3; i++) {
				is >> data;
				stringstream ss(data);
				while(getline(ss, token, '/')) {
					idx[i].push_back(stoi(token));
				}
			}
			Material * m = new Material(world);
			init_material_from_obj(m);
			Object * triangle = new Triangle(vertices[idx[0][0]], vertices[idx[1][0]], vertices[idx[2][0]], m);
			// cerr << "rendered\n";
			world->addObject(triangle);
		} else if(c == "v") {
			is >> v[0] >> v[1] >> v[2];
			vertices.push_back(Vector3D(v[0]*SCALING_FACTOR, v[1]*SCALING_FACTOR, v[2]*SCALING_FACTOR));
		} else if(c == "vn") {
			is >> v[0] >> v[1] >> v[2];
			normal_vertices.push_back(Vector3D(v[0], v[1], v[2]));
		} else if(c == "vt") {
			is >> v[0] >> v[1];
			texture_vertices.push_back(make_pair((double)v[0],(double)v[1]));
		}
	}
}

int init_resources(void)
{
	//Create program
	program = createProgram("vshader.vs", "fshader.fs");
	attribute_coord2d = glGetAttribLocation(program, "coord2d");
	if(attribute_coord2d == -1)
	{
		fprintf(stderr, "Could not bind location: coord2d\n");
		return 0;
	}

	//Initialize raytracer objects
	Vector3D camera_position(0, 10, 10);
	Vector3D camera_target(0, 0, 0); //Looking down -Z axis
	Vector3D camera_up(0, 1, 0);
	float camera_fovy =  45;
	camera = new Camera(camera_position, camera_target, camera_up, camera_fovy, screen_width, screen_height);
	//Create a world
	World *world = new World;
	world->setAmbient(Color(1));
	world->setBackground(Color(0.1, 0.3, 0.6));
	
	// Material *m = new Material(world);
	// m->color = Color(0.1, 0.7, 0.0);
	// m->ka = 0.2;
	// m->kd = 0.9;
	// m->ks = 0.4;
	// m->kr = 1.0;
	// m->kt = 0.0;
	// m->eta = 1.0;
	// m->n = 128;

	// Object *sphere = new Sphere(Vector3D(0, 0, 0), 3, m);
	// Material *m1 = new Material(world);
	// m1->color = Color(0.8, 0.1, 0.0);
	// m1->ka = 0.2;
	// m1->kd = 0.9;
	// m1->ks = 0.4;
	// m1->kr = 0.0;
	// m1->kt = 0.0;
	// m1->eta = 1.0;
	// m1->n = 128;

	// Material *m2 = new Material(world);
	// m2->color = Color(1.0, 1.0, 1.0);
	// m2->ka = 0.4;
	// m2->kd = 0.9;
	// m2->ks = 0.4;
	// m2->kr = 0.1;
	// m2->kt = 0.8;
	// m2->eta = 2.0;
	// m2->n = 128;
	// Object *cylinder = new Cylinder(Vector3D(-7,0,-3),1,Vector3D(0,0,1),m2);
	// Object *sphere2 = new Sphere(Vector3D(4,0,4), 3, m1);
	// Material *floorMat = new Material(world);
	// floorMat->color = Color(0.5, 0.5, 0.5);
	// floorMat->ka = 0.1;
	// floorMat->kd = 0.9;
	// floorMat->ks = 0.2;
	// floorMat->kt = 0.0;
	// floorMat->kr = 0.5;
	// floorMat->eta = 1.0;
	// Object *plane = new Plane(Vector3D(10, -3, 10), Vector3D(-10, -3, 10),Vector3D(-10, -3, -10),Vector3D(10, -3, -10), floorMat);
	// // world->addObject(sphere);
	 // world->addObject(sphere2);
	 // world->addObject(plane);
	 // world->addObject(cylinder);
	LightSource *light = new PointLightSource(world, Vector3D(-10, 10, 0), Color(1, 1, 1));
	//LightSource *light2 = new PointLightSource(world, Vector3D(0, 10, 0), Color(1, 1, 1));
	world->addLight(light);
	//world->addLight(light2);

	// load_image_from_obj(world, "pig_triangulated.obj");
	load_image_from_obj(world, "tetrahedron.obj");
	engine = new RenderEngine(world, camera);

	//Initialise texture
	glGenTextures(1, &texImage);
	glBindTexture(GL_TEXTURE_2D, texImage);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screen_width, screen_height, 0, GL_RGB, GL_UNSIGNED_BYTE, camera->getBitmap());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //Show pixels when zoomed
	uniform_texImage = glGetUniformLocation(program, "texImage");
	if(uniform_texImage == -1)
	{
		fprintf(stderr, "Could not bind uniform: texImage\n");
		return 0;
	}
	return 1;
}


void onDisplay()
{
    /* Clear the background as white */
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    
    //printOpenGLError();
    glUseProgram(program);
    glEnableVertexAttribArray(attribute_coord2d);
    glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, quadVertices);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texImage);
    glUniform1i(uniform_texImage, 0);

    glDrawArrays(GL_TRIANGLES, 0, 6);
    glDisableVertexAttribArray(attribute_coord2d);
    
    /* Display the result */
    glutSwapBuffers();
}

void free_resources()
{
    glDeleteProgram(program);
    glDeleteTextures(1, &texImage);
}

void onReshape(int width, int height) {
	screen_width = width;
	screen_height = height;
	glViewport(0, 0, screen_width, screen_height);
}

void SaveImage()
{
	ILuint imageID = ilGenImage();
	ilBindImage(imageID);
	ilTexImage(camera->getWidth(), camera->getHeight(), 1, 3, IL_RGB, IL_UNSIGNED_BYTE, camera->getBitmap());
	//ilEnable(IL_FILE_OVERWRITE);
	time_t rawtime;
	time (&rawtime);
	char time_str[26];
	ctime_r(&rawtime, time_str);
	time_str[strlen(time_str) - 1] =0;//Remove trailing return character.
	char imageName[256];
	sprintf(imageName, "Lumina %s.png", time_str);
	ilSave(IL_PNG, imageName);
	fprintf(stderr, "Image saved as: %s\n", imageName);
}

void onKey(unsigned char key, int x, int y)
{
	switch(key)
	{
		case 27: exit(0);
		break;
		case 's': //Save to image
		case 'S': //Save to image
			SaveImage();
		break;
		
	}
}

void onIdle(void)
{
	static bool done = false;
	//Generate a pretty picture
	if(!done)
	{
		if(engine->renderLoop())
		{
			done = true;
			fprintf(stderr, "Rendering complete.\n");
		}
	
		//Update texture on GPU
		glBindTexture(GL_TEXTURE_2D, texImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screen_width, screen_height, GL_RGB, GL_UNSIGNED_BYTE, camera->getBitmap());

		glutPostRedisplay();
	}
}

int main(int argc, char* argv[])
{
	if(argc > 1)
	{
		screen_width = atoi(argv[1]);
		screen_height = atoi(argv[2]);
		screen_width -= (screen_width % 2); //Make it even
		screen_height -= (screen_height % 2); //Make it even
	}
	fprintf(stderr, "Welcome to Lumina raytracer.\nFull command: %s [width] [height]\nPress 's' to save framebufer to disk.\n", argv[0]);
	/* Glut-related initialising functions */
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
	glutInitWindowSize(screen_width, screen_height);
	glutCreateWindow("Assignment 6: Raytracing");
#ifndef __APPLE__
	GLenum glew_status = glewInit();
	if(glew_status != GLEW_OK)
	{
		fprintf(stderr, "Error: %s\n", glewGetErrorString(glew_status));
		return EXIT_FAILURE;
	}
#endif

	ilInit();


	/* When all init functions run without errors,
	   the program can initialise the resources */
	if (1 == init_resources())
	{
		/* We can display it if everything goes OK */
		glutReshapeFunc(onReshape);
		glutDisplayFunc(onDisplay);
		glutKeyboardFunc(onKey);
		glutIdleFunc(onIdle);
		glutMainLoop();
	}

	/* If the program exits in the usual way,
	   free resources and exit with a success */
	free_resources();
	return EXIT_SUCCESS;
}
