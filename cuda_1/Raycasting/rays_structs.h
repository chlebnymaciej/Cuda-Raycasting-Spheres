#pragma once

#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#define radianCalc 3.141596f/ 180.0f

// struktura sphere po stronie hosta
struct Spheres {
    float3* center;
    float* radius;
    uchar3* color;
};
// SoA dla device
struct SphereSoA {
    float* x;
    float* y;
    float* z;
    float* r;
    unsigned char* R;
    unsigned char* G;
    unsigned char* B;
 };

// sfera dla pamieci constant
struct Sphere {
    float4 center;
    uchar3 color;
};

// struktura swiatla
struct Lights {
    float3* position;
    uchar3* color;
};


enum Mode {
    GPURayCastingWithShadowsConstantMemory,
    GPURayCastingConstantMemory,
    GPURayCastingWithShadowsGlobalMemory,
    GPURayCastingGlobalMemory,
    CPURayCastingWithShadows,
    CPURayCasting
};

enum SphereDistribution {
    Rows,
    Uniform,
    Uniform3d
};

// kamera z wyliczeniami
struct Camera
{
    float vfov;
    float r = 1.0f;
    float theta;
    float fi;
    float3 look_from;
    float3 vup;
    Camera()
    {
        vfov = 0.5;
        look_from = { 400,150,0 };
        theta = 0;
        fi = -90;
        vup = { 0,1,0 };
    }

    // funkcja wyliczajace punkt na ktory patrzymy
    __host__ __device__ float3 look_atf()
    {
        const float x = look_from.x + r * cosf(theta * radianCalc) * cosf(fi * radianCalc);
        const float z = look_from.z + r* cosf(theta * radianCalc) * sinf(fi * radianCalc);
        const float y = look_from.y + r * sinf(theta * radianCalc);
        float3 t = { x,y,z };
        return t;
    }

};

// wartosci parametrow okna przez ktore patrzymy
struct Screen2 {
    float3 eye_point;
    float3 horizontal;
    float3 vertical;
    float3 lower_left_corner;
};