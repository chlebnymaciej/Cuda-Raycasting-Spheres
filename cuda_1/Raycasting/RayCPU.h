#pragma once
#include <vector_types.h>
#include <rays_structs.h>


void rayCastingCPU(uchar4* dst, const int imageW, const int imageH, Spheres spheresD,
	int len, Lights lights_D, int len_lights, Camera cam, Mode mode);