#ifndef _RAYCASTING_KERNEL_h_
#define _RAYCASTING_KERNEL_h_

#include <vector_types.h>
#include "rays_structs.h"

extern "C" void rayCasting(uchar4 * dst, const int imageW, const int imageH, SphereSoA spheresD,
	int len, Lights lights_D, int len_lights, Camera cam, Mode mode);

extern "C" void rayCastingCPU(uchar4* dst, const int imageW, const int imageH, Spheres spheresD,
    int len, Lights lights_D, int len_lights, Camera camera, Mode mode);

extern "C" void CopyToConstantData(Spheres spheres, int len);

#endif
