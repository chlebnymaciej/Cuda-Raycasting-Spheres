#pragma once
#include <helper_math.h>
#include <helper_gl.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/wglew.h>
#endif

#include <GL/freeglut.h>

// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <chrono>

#include "RayKernel.cuh"
#include "rays_structs.h"


static const char* _cudaGetErrorEnum(cudaError_t error) {
	return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const* const func, const char* const file,
	int const line) {
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		exit(EXIT_FAILURE);
	}
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex, gl_Shader;
struct cudaGraphicsResource* cuda_pbo_resource; // handles OpenGL-CUDA exchange

//Source image on the host side
uchar4* h_Src = 0;
// Destination image on the GPU side
uchar4* d_dst = NULL;

//Original image width and height
int imageW = 800, imageH = 400;

float t = 2.5;

#define Move_delta 10.0f
#define REFRESH_DELAY 10 //ms
#define BUFFER_DATA(i) ((char *)0 + i)
#define ANGLE_DELTA 1.0f
#define SPHERES 1000
Spheres* sphere_host = NULL;
Spheres spheresD;
SphereSoA spheresDSoA;
Lights* lights_host = NULL;
Lights lights_D;
float* lights_orbit = NULL;
float* lights_angles = NULL;
int len;
int len_lights;
int max_lights;


Camera* camera = NULL;
bool moving_lights = true;
Mode mode = GPURayCastingConstantMemory;
SphereDistribution dist = Uniform;

void ChangeSpheres();

void renderImage()
{
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_dst, &num_bytes, cuda_pbo_resource));
	
	if(mode == CPURayCasting || mode == CPURayCastingWithShadows)
		rayCastingCPU(d_dst, imageW, imageH, *sphere_host, len, *lights_host, len_lights, *camera, mode);
	else
		rayCasting(d_dst, imageW, imageH, spheresDSoA, len, lights_D, len_lights, *camera, mode);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

 // OpenGL display function
void displayFunc(void)
{
	renderImage();

	// load texture from PBO
	//  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));

	// fragment program is required to display floating point texture
	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_Shader);
	glEnable(GL_FRAGMENT_PROGRAM_ARB);
	glDisable(GL_DEPTH_TEST);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(0.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(1.0f, 1.0f);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(0.0f, 1.0f);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_FRAGMENT_PROGRAM_ARB);

	glutSwapBuffers();
} // displayFunc



void cleanup()
{
	if (h_Src)
	{
		delete[] h_Src;
		h_Src = 0;
	}

	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glDeleteBuffers(1, &gl_PBO);
	glDeleteTextures(1, &gl_Tex);
	glDeleteProgramsARB(1, &gl_Shader);
}

void initMenus()
{

	printf("h - help\n");
	printf("w, s, a, d - poruszanie\n");
	printf("i, k, j, l - obrot kamer¹\n");
	printf("c, v - poruszanie w gore i w dol\n");
	printf("t - wylacz / wlacz poruszanie sie swiatel\n");
	printf("1, 2, 3, 4, 5, 6 - przelaczanie miedzy implementacjami\n");
	printf("7, 8, 9 - zmiana rodzaju dystrybucji sfer\n");
	printf("-, = -odpowiednio zmniejsz / zwieksz liczbe swiatel\n");
}

void setTitle()
{
	char title[256];
	char* modeString;
	switch (mode)
	{
	case GPURayCastingWithShadowsConstantMemory:
		modeString = "GPU Constant Shadow";
		break;
	case GPURayCastingConstantMemory:
		modeString = "GPU Constant";
		break;
	case GPURayCastingWithShadowsGlobalMemory:
		modeString = "GPU Global Shadow";
		break;
	case GPURayCastingGlobalMemory:
		modeString = "GPU Global";
		break;
	case CPURayCastingWithShadows:
		modeString = "CPU Shadow";
		break;
	case CPURayCasting:
		modeString = "CPU";
		break;
	}
	sprintf(title, "Mode: %s, lights: %d, %d Fi %d Theta,", 
		modeString,
		len_lights,
		static_cast<int>(camera->fi),
		static_cast<int>(camera->theta));
	glutSetWindowTitle(title);
}
// OpenGL keyboard function
void keyboardFunc(unsigned char k, int, int)
{
	switch (k)
	{
	case '\033':
	case 'q':
	case 'Q':
		printf("Shutting down...\n");
		glutDestroyWindow(glutGetWindow());
		return;
	case 'W':
	case 'w':
		camera->look_from = 10.0f * (camera->look_atf() - camera->look_from) + camera->look_from;
		break;
	case 'S':
	case 's':
		camera->look_from = -10.0f * (camera->look_atf() - camera->look_from) + camera->look_from;
		break;
	case 'A':
	case 'a':
	{
		const float3 eye_point = camera->look_from;
		const float3 z_wersor = normalize(camera->look_from - camera->look_atf());
		const float3 x_wersor = normalize(cross(camera->vup, z_wersor));
	
		camera->look_from = -10.0f * x_wersor + camera->look_from;
	}
		break;
	case 'D':
	case 'd':
	{
		const float3 eye_point = camera->look_from;
		const float3 z_wersor = normalize(camera->look_from - camera->look_atf());
		const float3 x_wersor = normalize(cross(camera->vup, z_wersor));

		camera->look_from = 10.0f * x_wersor + camera->look_from;
	}
		break;
	case 'C':
	case 'c':
		camera->look_from.y += Move_delta;

		break;
	case 'V':
	case 'v':
		camera->look_from.y -= Move_delta;

		break;
	case 'j':
	case 'J':
		if (camera->fi == -180)
			camera->fi = 180;
		camera->fi -= ANGLE_DELTA;
		break;
	case 'l':
	case 'L':
		if (camera->fi == 180)
			camera->fi = -180;
		camera->fi += ANGLE_DELTA;
		break;
	case 'i':
	case 'I':
		if (camera->theta < 89.0f)
			camera->theta += ANGLE_DELTA;
		break;
	case 'k':
	case 'K':
		if (camera->theta > -89.0f)
			camera->theta -= ANGLE_DELTA;
		break;
	case 'T':
	case 't':
		moving_lights = !moving_lights;
		break;
	case 'H':
	case 'h':
		initMenus();
		break;
	case '1':
		mode = GPURayCastingWithShadowsConstantMemory;
		break;
	case '2':
		mode = GPURayCastingConstantMemory;
		break;
	case '3':
		mode = GPURayCastingWithShadowsGlobalMemory;
		break;
	case '4':
		mode = GPURayCastingGlobalMemory;
		break;
	case '5':
		mode = CPURayCastingWithShadows;
		break;
	case '6':
		mode = CPURayCasting;
		break;
	case '7':
		dist = Rows;
		ChangeSpheres();
		break;
	case '8':
		dist = Uniform;
		ChangeSpheres();
		break;
	case '9':
		dist = Uniform3d;
		ChangeSpheres();
		break;
	case '-':
		if (len_lights > 1) len_lights -= 1;
		break;
	case '=':
		if (len_lights < max_lights) len_lights += 1;
		break;
	default:
		break;
	}
	setTitle();
} // keyboardFunc

void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}

	if (moving_lights)
	{
		// obracanie swiatel wokol orbit 
		auto lambdaRadians = [](float angle) {return angle / 180.0f * 3.1415f; };
		float delta = 0.25f;
		for (int i = 0; i < len_lights; i++)
		{
			float radius = lights_orbit[i];
			float fi = lights_angles[i];
			fi += delta;

			float x = radius * cosf(lambdaRadians(fi)) + 400.0f;
			float z = radius * sinf(lambdaRadians(fi)) - 500.0f;

			lights_host->position[i].x = x;
			lights_host->position[i].z = z;
			lights_angles[i] = fi;
		}
		cudaMemcpy(lights_D.position, lights_host->position, sizeof(float3) * len_lights, cudaMemcpyHostToDevice);
	}
}

// gl_Shader for displaying floating-point texture
static const char* shader_code =
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";


GLuint compileASMShader(GLenum program_type, const char* code)
{
	GLuint program_id;
	glGenProgramsARB(1, &program_id);
	glBindProgramARB(program_type, program_id);
	glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, static_cast<GLsizei>(strlen(code)), (GLubyte*)code);

	GLint error_pos;
	glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

	if (error_pos != -1)
	{
		const GLubyte* error_string;
		error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
		fprintf(stderr, "Program error at position: %d\n%s\n", static_cast<int>(error_pos), error_string);
		return 0;
	}

	return program_id;
}

void initOpenGLBuffers(int w, int h)
{
	if (h_Src)
	{
		delete[] h_Src;
		h_Src = 0;
	}

	if (gl_Tex)
	{
		glDeleteTextures(1, &gl_Tex);
		gl_Tex = 0;
	}

	if (gl_PBO)
	{
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &gl_PBO);
		gl_PBO = 0;
	}

	// allocate new buffers
	h_Src = new uchar4[w * h];

	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);

	glGenBuffers(1, &gl_PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_Src, GL_STREAM_COPY);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO,
		cudaGraphicsMapFlagsWriteDiscard));

	// load shader program
	gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void reshapeFunc(int w, int h)
{
	glViewport(0, 0, w, h);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

	if (w != 0 && h != 0)  // Do not call when window is minimized that is when width && height == 0
		initOpenGLBuffers(w, h);

	imageW = w;
	imageH = h;
	glutPostRedisplay();
}

void initGL(int* argc, char** argv)
{
	printf("Initializing GLUT...\n");
	glutInit(argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(imageW, imageH);
	glutInitWindowPosition(0, 0);
	glutCreateWindow(argv[0]);

	glutDisplayFunc(displayFunc);
	glutKeyboardFunc(keyboardFunc);
	glutReshapeFunc(reshapeFunc);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	if (!isGLVersionSupported(1, 5) ||
		!areGLExtensionsSupported("GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
	{
		fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
		exit(EXIT_SUCCESS);
	}

	printf("OpenGL window created.\n");
}

// ////////////////////////////////////////////////
// Funkcje inicjuj¹ce dane
// ////////////////////////////////////////////////
void InitLights(Lights* lights, int* lenp) {
	if (lenp == NULL) return;
	int length = 10;
	lights->color = new uchar3[length];
	lights->position = new float3[length];

	lights_angles = new float[length];
	lights_orbit = new float[length];

	float a = 5000.0f;
	float fi = 90;
	float x, y=1000.0f, z, radius;
	auto lambdaRadians = [](float angle) {return angle / 180.0f * 3.1415f; };
	for (int i = 0; i < length; i++)
	{
		radius = lambdaRadians(fi)*a;
		y += 150.f;
		x = radius * cosf(lambdaRadians(fi)) + 400.0f;
		z = radius * sinf(lambdaRadians(fi)) - 500.0f;
		fi += 45.0f;
		lights->color[i] = { 255,255,255 };
		lights->position[i] = { x,y,z };

		lights_angles[i] = fi;
		lights_orbit[i] = radius;
	}
	*lenp = length;
}

void InitSpheres(Spheres* spheres, int* lenp) {
	if (lenp == NULL) return;
	int length = SPHERES;
	spheres->center = new float3[length];
	spheres->color = new uchar3[length];
	spheres->radius = new float[length];

	srand(static_cast<unsigned int>(time(NULL)));
	int i = 0;
	for (int z = -300; i < SPHERES; z -= 150)
	{
		for (int x = -800; x < 1200; x += 200)
		{
			spheres->radius[i] = 50.0f;
			spheres->center[i] = { (float)x, 65.0f, (float)z };
			spheres->color[i] = { (unsigned char)(rand() % 240 + 10),
								  (unsigned char)(rand() % 240 + 10),
								  (unsigned char)(rand() % 240 + 10) };
			i++;
		}
	}

	spheres->radius[2] = 10000000.0f;
	spheres->color[2] = { 192,192,192 };
	spheres->center[2] = { 400,-10000000.0f + 20.0,-550 };
	*lenp = length;
	printf("Kule %d\n", length);
}
void InitSpheresUniform3D(Spheres* spheres, int* lenp) {
	if (lenp == NULL) return;
	int lengths = SPHERES;

	spheres->center = new float3[lengths];
	spheres->color = new uchar3[lengths];
	spheres->radius = new float[lengths];

	srand(static_cast<unsigned int>(time(NULL)));	int i = 0;
	int max = 10000;
	int max_radius = 500;
	bool why = false;
	while (i < lengths)
	{

		float x = (float)(rand() % max);
		float r = (float)(rand() % max_radius);
		float y = (float)(rand() % max) + r;
		float z = -(float)(rand() % max);
		why = false;
		float3 tmp = { x,y,z };
		for (int j = 0; j < i; j++)
		{
			if (length(tmp - spheres->center[j]) < (spheres->radius[j] + r))
			{
				why = true;
				break;
			}
		}
		if (!why)
		{
			spheres->radius[i] = r;
			spheres->center[i] = tmp;
			spheres->color[i] = { (unsigned char)(rand() % 240 + 10),
							  (unsigned char)(rand() % 240 + 10),
							  (unsigned char)(rand() % 240 + 10) };
			i++;
		}
	}
	
	spheres->radius[2] = 10000000.0f;
	spheres->color[2] = { 192,192,192 };
	spheres->center[2] = { 400,-10000000.0f,-550 };

	*lenp = lengths;
	printf("Kule %d\n", lengths);
}
void InitSpheresUniform(Spheres* spheres, int* lenp) {
	if (lenp == NULL) return;
	int lengths = SPHERES;
	spheres->center = new float3[lengths];
	spheres->color = new uchar3[lengths];
	spheres->radius = new float[lengths];

	srand(static_cast<unsigned int>(time(NULL)));
	int i = 0;
	int max = 10000;
	int max_radius = 200;
	bool why = false;
	while (i < lengths)
	{

		float x = (float)(rand() % max + 50);
		float z = -(float)(rand() % max - 50);
		float r = (float)(rand() % max_radius);
		float y = r;

		why = false;
		float3 tmp = { x,y,z };
		for (int j = 0; j < i; j++)
		{
			if (length(tmp - spheres->center[j]) < (spheres->radius[j] + r))
			{
				why = true;
				break;
			}
		}
		if (!why)
		{
			spheres->radius[i] = r;
			spheres->center[i] = tmp;
			spheres->color[i] = { (unsigned char)(rand() % 240 + 10),
							  (unsigned char)(rand() % 240 + 10),
							  (unsigned char)(rand() % 240 + 10) };
			i++;
		}
	}
	spheres->radius[2] = 10000000.0f;
	spheres->color[2] = { 192,192,192 };
	spheres->center[2] = { 400,-10000000.0f,-550 };

	*lenp = lengths;
	printf("Kule %d\n", lengths);
}

void CopyToDevice()
{
	CopyToConstantData(*sphere_host, len);
	cudaMemcpy(lights_D.position, lights_host->position, sizeof(float3) * len_lights, cudaMemcpyHostToDevice);
	cudaMemcpy(lights_D.color, lights_host->color, sizeof(uchar3) * len_lights, cudaMemcpyHostToDevice);
	float* tmp = new float[len];
	for (int i = 0; i < len; i++)
	{
		tmp[i] = sphere_host->center[i].x;
	}
	cudaMemcpy(spheresDSoA.x, tmp, sizeof(float) * len, cudaMemcpyHostToDevice);

	for (int i = 0; i < len; i++)
	{
		tmp[i] = sphere_host->center[i].y;
	}
	cudaMemcpy(spheresDSoA.y, tmp, sizeof(float) * len, cudaMemcpyHostToDevice);

	for (int i = 0; i < len; i++)
	{
		tmp[i] = sphere_host->center[i].z;
	}
	cudaMemcpy(spheresDSoA.z, tmp, sizeof(float) * len, cudaMemcpyHostToDevice);
	cudaMemcpy(spheresDSoA.r, sphere_host->radius, sizeof(float) * len, cudaMemcpyHostToDevice);

	delete[] tmp;
	unsigned char* tmpC = new unsigned char[len];

	for (int i = 0; i < len; i++)
	{
		tmpC[i] = sphere_host->color[i].x;
	}
	cudaMemcpy(spheresDSoA.R, tmpC, sizeof(unsigned char) * len, cudaMemcpyHostToDevice);

	for (int i = 0; i < len; i++)
	{
		tmpC[i] = sphere_host->color[i].y;
	}
	cudaMemcpy(spheresDSoA.G, tmpC, sizeof(unsigned char) * len, cudaMemcpyHostToDevice);

	for (int i = 0; i < len; i++)
	{
		tmpC[i] = sphere_host->color[i].z;
	}
	cudaMemcpy(spheresDSoA.B, tmpC, sizeof(unsigned char) * len, cudaMemcpyHostToDevice);
	delete[] tmpC;

}

void ChangeSpheres()
{
	delete[] sphere_host->center;
	delete[] sphere_host->color;
	delete[] sphere_host->radius;

	switch (dist)
	{
	case Rows:
		InitSpheres(sphere_host, &len);
		break;
	case Uniform:
		InitSpheresUniform(sphere_host, &len);
		break;
	case Uniform3d:
		InitSpheresUniform3D(sphere_host, &len);
		break;
	default:
		break;
	}

	CopyToDevice();
}

void InitData()
{
	camera = new Camera();
	sphere_host = new Spheres();
	InitSpheresUniform3D(sphere_host, &len);

	lights_host = new Lights();
	InitLights(lights_host, &len_lights);
	max_lights = len_lights;

	cudaMalloc(&(lights_D.position), sizeof(float3) * len_lights);
	cudaMalloc(&(lights_D.color), sizeof(uchar3) * len_lights);

	cudaMalloc(&(spheresDSoA.x), sizeof(float) * len);
	cudaMalloc(&(spheresDSoA.y), sizeof(float) * len);
	cudaMalloc(&(spheresDSoA.z), sizeof(float) * len);
	cudaMalloc(&(spheresDSoA.r), sizeof(float) * len);
	cudaMalloc(&(spheresDSoA.R), sizeof(unsigned char) * len);
	cudaMalloc(&(spheresDSoA.G), sizeof(unsigned char) * len);
	cudaMalloc(&(spheresDSoA.B), sizeof(unsigned char) * len);

	CopyToDevice();
}
void DeleteData()
{
	cudaFree(lights_D.color);
	cudaFree(lights_D.position);

	delete[] lights_host->position;
	delete[] lights_host->color;
	delete lights_host;

	cudaFree(spheresDSoA.x);
	cudaFree(spheresDSoA.y);
	cudaFree(spheresDSoA.z);
	cudaFree(spheresDSoA.r);
	cudaFree(spheresDSoA.R);
	cudaFree(spheresDSoA.G);
	cudaFree(spheresDSoA.B);

	delete[] sphere_host->center;
	delete[] sphere_host->color;
	delete[] sphere_host->radius;
	delete sphere_host;
}




////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
#endif	
	initGL(&argc, argv);
	initOpenGLBuffers(imageW, imageH);
	InitData();

	glutCloseFunc(cleanup);

	glutMainLoop();

	DeleteData();
} // main
