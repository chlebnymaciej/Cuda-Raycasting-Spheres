#include "rays_structs.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "RayKernel.cuh"
#include <helper_math.h>
#include <stdio.h>


#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16
#define BLOCKS(x, dim) (x / dim + (x % dim == 0 ? 0 : 1))
#define MAX_SPHERES_IN_CONSTANT_MEMORY 2000

__constant__ __device__ Sphere spheres_constant[MAX_SPHERES_IN_CONSTANT_MEMORY];

__host__ __device__ inline float3 make_colorf(uchar3 u)
{
    float3 col = { static_cast<float>(u.x), static_cast<float>(u.y), static_cast<float>(u.z) };
    return col;
}

__host__ __device__ inline uchar3 make_color(float3 u)
{
    float3 c = clamp(u, 0, 255);
    uchar3 col = { static_cast<unsigned char>(c.x),static_cast<unsigned char>(c.y),static_cast<unsigned char>(c.z) };
    return col;
}



////////////////////////////////////////////////////////////////////////////////
// Functions that gets sphere data for different memory type
////////////////////////////////////////////////////////////////////////////////
__device__ void GetSphereConstantMemory(SphereSoA spheres, int i, float3* center, float* radius)
{
    *center = make_float3(spheres_constant[i].center);
    *radius = spheres_constant[i].center.w;
}

__device__ void GetSphereGlobalMemory(SphereSoA spheres, int i, float3* center, float* radius)
{
    *center = { spheres.x[i],spheres.y[i], spheres.z[i] };
    *radius = spheres.r[i];
}

__device__ void GetColorConstantMemory(SphereSoA spheres, int i, float3* color)
{
    *color = make_colorf(spheres_constant[i].color);
}

__device__ void GetColorGlobalMemory(SphereSoA spheres, int i, float3* color)
{
    *color = { static_cast<float>(spheres.R[i]),
               static_cast<float>(spheres.G[i]), 
               static_cast<float>(spheres.B[i])
    };
}

__device__ void GetSphere(SphereSoA spheres, int i, Mode mode, float3* center, float* radius)
{
    switch (mode)
    {
    case GPURayCastingWithShadowsConstantMemory:
    case GPURayCastingConstantMemory:
        GetSphereConstantMemory(spheres, i, center, radius);
        break;
    case GPURayCastingWithShadowsGlobalMemory:
    case GPURayCastingGlobalMemory:
        GetSphereGlobalMemory(spheres, i, center, radius);
        break;
    }
}

__device__ void GetSphereColor(SphereSoA spheres, int i, Mode mode, float3* color)
{
    switch (mode)
    {
    case GPURayCastingWithShadowsConstantMemory:
    case GPURayCastingConstantMemory:
        GetColorConstantMemory(spheres, i, color);
        break;
    case GPURayCastingWithShadowsGlobalMemory:
    case GPURayCastingGlobalMemory:
        GetColorGlobalMemory(spheres, i, color);
        break;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Function that calculates if rays hits light directly
////////////////////////////////////////////////////////////////////////////////
__host__ __device__ int CalculateSun(Lights lights, int len, float3 eye_point, float3 direction,
    float3* intersection, float* closest_distance)
{
    int any = -1;
    // iteracja po zrodlach swiatla
    for (int i = 0; i < len; i++) {
        float3 center = lights.position[i];
        // promien zrodla swiatla na potrzeby wizualizacji
        float radius = 20.0f;

        float3 originCenter = eye_point - center;
        float a = dot(direction, direction);
        float b = 2.0f * dot(originCenter, direction);
        float c = dot(originCenter, originCenter) - radius * radius;
        float delta = b * b - 4 * a * c;
        float dist_current = -1;
        float3 intersection_current = { 0,0,0 };

        // wyliczenie na podstawie rownania kwadratowego
        if (delta < 0) continue;
        else if (delta > 0)
        {
            float t1 = (-b + sqrtf(delta)) / (2.0f * a);
            float t2 = (-b - sqrtf(delta)) / (2.0f * a);
            float tmin = min(t1, t2);
            float tmax = max(t1, t2);
            if (tmin > 0.001f)
            {
                intersection_current = eye_point + tmin * direction;
            }
            else if (tmin < 0.001f && tmax < 0.001f) continue;
            else
            {
                intersection_current = eye_point + tmax * direction;
            }

        }
        else
        {
            float t = -b / (2.0f * a);
            if (t < 0.001f) continue;
            intersection_current = eye_point + t * direction;
        }
        dist_current = length(intersection_current - eye_point);
        if (dist_current < *closest_distance || *closest_distance < 0.001)
        {
            *closest_distance = dist_current;
            *intersection = intersection_current;
            any = i;
        }
    }
    return any;
}



////////////////////////////////////////////////////////////////////////////////
// Function that calculates closest spehere to the viewer
////////////////////////////////////////////////////////////////////////////////
__device__ int CalculateHitToClosest(SphereSoA spheres,
    int len,
    float3 eye_point,
    float3 direction,
    float3* intersection,
    float* closest_distance,
    Mode mode,
    bool lightMode = false)
{
    int any = -1;
    // iteracja po kazdej sferze
    for (int i = 0; i < len; i++) {

        float3 center;
        float radius;
        // pobranie wartosci paramtrow sfery
        GetSphere(spheres, i, mode, &center, &radius);

        float3 originCenter = eye_point - center;
        float a = dot(direction, direction);
        float b = 2.0f * dot(originCenter, direction);
        float c = dot(originCenter, originCenter) - radius * radius;
        // wyliczenie przeciecia z rownania kwadratowego
        float delta = b * b - 4 * a * c;
        float dist_current = -1;
        float t_local = -1;
        float3 intersection_current = { 0,0,0 };

        // brak przeciecia
        if (delta < 0) continue;
        // przeciecie w dwoch miejscach
        else if (delta > 0)
        {
            float t1 = (-b + sqrtf(delta)) / (2.0f * a);
            float t2 = (-b - sqrtf(delta)) / (2.0f * a);
            float tmin = min(t1, t2);
            float tmax = max(t1, t2);
            // bierzemy pierwszy wartosc wieksza niz 0 (w naszym przypadku 0.001)
            if (tmin > 0.001f)
            {
                intersection_current = eye_point + tmin * direction;
                t_local = tmin;
            }
            else if (tmin < 0.001f && tmax < 0.001f) continue;
            else
            {
                intersection_current = eye_point + tmax * direction;
                t_local = tmax;
            }

        }
        // przeciecie w jednym miejscu
        else
        {
            float t = -b / (2.0f * a);
            if (t < 0.001f) continue;
            intersection_current = eye_point + t * direction;
            t_local = t;
        }
        dist_current = length(intersection_current - eye_point);
        // wyznaczenie nowej najblizszej sfery 
        if (dist_current < *closest_distance || *closest_distance < 0.001)
        {
            *closest_distance = dist_current;
            *intersection = intersection_current;
            any = i;
            // sprawdzenie przeciecia dla cieni
            if (lightMode && t_local<1 && t_local>-1)
                return 1;
        }
    }
    return any;
}

////////////////////////////////////////////////////////////////////////////////
// Function that calculates pixel color
////////////////////////////////////////////////////////////////////////////////
__global__ void RayCastingSpheres(
    uchar4* dst,
    const int imageW,
    const int imageH,
    SphereSoA spheres,
    const int spheres_len,
    Lights lights,
    const int lights_len,
    Screen2 screen,
    Mode mode,
    bool shadow = false
)
{
    // wyliczenie wspolrzednych piksela
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= imageW || y >= imageH) return;

   
    float3 eye_point = screen.eye_point;

    // wektor kierunku promienia
    float3 direction = screen.lower_left_corner
        + ((float)y / imageH) * screen.vertical
        + ((float)x / imageW) * screen.horizontal - eye_point;

    // indeks i odleglosc do najblizszej sfery (-1 - promien nie uderza w sfere)
    int i = -1;
    float closest_distance = -1;
    // punkt uderzenia promienia w sfere
    float3 intersection;

    // funkcja obliczajaca najblizsza sfere
    i = CalculateHitToClosest(spheres, spheres_len,
        eye_point, direction,
        &intersection, &closest_distance,
        mode);

    // jesli promien nie uderzyl zadnej sfery
    if (i<0 || closest_distance < 0.001f)
    {
        float3 interLight;
        float dist_light = -1;

        // wyliczenie i wizualizacja polozenia swiatla
        int light_i = CalculateSun(lights, lights_len, eye_point, direction, &interLight, &dist_light);
        if (light_i > 0)
        {
            dst[y * imageW + x] = { lights.color[light_i].x,lights.color[light_i].y,lights.color[light_i].z,0 };
            return;
        }

        // wyliczenia koloru tla
        float t = ((float)y) / imageH;
        float3 start = { 255.0,255.0,255.0 };
        float3 end = { 0.5 * 255.0,0.7 * 255.0,255.0 };
        float3 col = lerp(start, end, t);
        dst[y * imageW + x] = { 
            static_cast<unsigned char>(col.x),
            static_cast<unsigned char>(col.y),
            static_cast<unsigned char>(col.z),
            0 
        };
        return;
    }

    float3 sphere_center;
    float sphere_radius;
    GetSphere(spheres, i, mode, &sphere_center, &sphere_radius);

    // wektor normalny do sfery w punkcie uderzenia promienia
    float3 normalVector = normalize(intersection - sphere_center);
    // wspolczynniki poczatkowe kolorow
    float3 color = { 0,0,0 };
    // kolor sfery
    float3 sphere_color;
    GetSphereColor(spheres, i, mode, &sphere_color);

    // wyliczenia koloru dla kazdego swiatla jako skladowe modelu Phonga
    for (int j = 0; j < lights_len; j++)
    {
        float closest_distance_light = -1;
        float3 intersection_light;
        float3 direction_light = lights.position[j] - intersection;

        // wyliczenie cieni (opcjonalne)
        if (shadow)
        {
            // sprawdzenie czy zaden element nie rzuca cienia od danego swiatla na dany punkt sfery
            int inters = CalculateHitToClosest(spheres, spheres_len,
                intersection, direction_light,
                &intersection_light, &closest_distance_light, mode, true);

            // jesli tak to nie dodajemy skladowych od danego swiatla
            if (inters > -1) continue;
        }

        // wyliczenie skladowych swiatla
        float3 light_color = make_colorf(lights.color[j]);
        direction_light = normalize(direction_light);
        float dotNL = clamp(dot(normalVector, direction_light),0.0f,1.0f);
        float3 r = normalize(2.0f * dotNL * direction_light - normalVector);
        float3 v = normalize(eye_point -intersection);

        float diffuse = clamp(dotNL, 0.0f, 1.0f);
        float specular = clamp(powf(fmaxf(dot(r, v),0.0f), 64), 0.0f, 1.0f);
        color += (specular + diffuse) * light_color / 255.0f;
    }
    // ograniczenie kolorow w zakresie RGB (0-255)
    color = clamp(sphere_color * color, 0, 255);
    dst[y * imageW + x] = { 
                static_cast<unsigned char>(color.x),
                static_cast<unsigned char>(color.y),
                static_cast<unsigned char>(color.z),
                0
    };
}
////////////////////////////////////////////////////////////////////////////////
// Function that calculates window coordinates
////////////////////////////////////////////////////////////////////////////////
__host__ __device__ Screen2 CalculateScreen(Camera cam, int imageW, int imageH)
{
    // Obliczamy wycinek plaszczyzny odpowiadaj¹cy naszemu oknu

    // kat jaki widzimy w pionie od srodka do gronej krawedzi
    float theta = cam.vfov; 
    float h = tanf(theta / 2.0f);
    // przeskalowana wysokosc
    float viewport_height = 2.0f * h; 
    // przeskalowana szerosc z zachowaniem proporcji ekranu
    float viewport_width = (imageW * viewport_height) / imageH; 

    // wirtualny punkt z ktorego wychodza promienie
    float3 eye_point = cam.look_from; 

    // przesuniecie plaszczyzny po osiach
    float3 z_wersor = normalize(cam.look_from - cam.look_atf());
    float3 x_wersor = normalize(cross(cam.vup, z_wersor));
    float3 y_wersor = cross(z_wersor, x_wersor);

    float3 horizontal = viewport_width * x_wersor;
    float3 vertical = viewport_height * y_wersor;
    float3 lower_left_corner = eye_point - horizontal / 2 - vertical / 2 - z_wersor;

    Screen2 screen = { eye_point, horizontal, vertical, lower_left_corner };
    return screen;
}

////////////////////////////////////////////////////////////////////////////////
// CPU Version
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Function that calculates if rays hits light directly
////////////////////////////////////////////////////////////////////////////////
//__host__ int CalculateSunCPU(Lights lights, int len, float3 eye_point, float3 direction,
//    float3* intersection, float* closest_distance)
//{
//    int any = -1;
//    // iteracja po zrodlach swiatla
//    for (int i = 0; i < len; i++) {
//        float3 center = lights.position[i];
//        // promien zrodla swiatla na potrzeby wizualizacji
//        float radius = 20.0f;
//
//        float3 originCenter = eye_point - center;
//        float a = dot(direction, direction);
//        float b = 2.0f * dot(originCenter, direction);
//        float c = dot(originCenter, originCenter) - radius * radius;
//        float delta = b * b - 4 * a * c;
//        float dist_current = -1;
//        float3 intersection_current = { 0,0,0 };
//
//        // wyliczenie na podstawie rownania kwadratowego
//        if (delta < 0) continue;
//        else if (delta > 0)
//        {
//            float t1 = (-b + sqrtf(delta)) / (2.0f * a);
//            float t2 = (-b - sqrtf(delta)) / (2.0f * a);
//            float tmin = min(t1, t2);
//            float tmax = max(t1, t2);
//            if (tmin > 0.001f)
//            {
//                intersection_current = eye_point + tmin * direction;
//            }
//            else if (tmin < 0.001f && tmax < 0.001f) continue;
//            else
//            {
//                intersection_current = eye_point + tmax * direction;
//            }
//
//        }
//        else
//        {
//            float t = -b / (2.0f * a);
//            if (t < 0.001f) continue;
//            intersection_current = eye_point + t * direction;
//        }
//        dist_current = length(intersection_current - eye_point);
//        if (dist_current < *closest_distance || *closest_distance < 0.001)
//        {
//            *closest_distance = dist_current;
//            *intersection = intersection_current;
//            any = i;
//        }
//    }
//    return any;
//}

////////////////////////////////////////////////////////////////////////////////
// Function that calculates closest spehere to the viewer
////////////////////////////////////////////////////////////////////////////////
__host__ int CalculateHitToClosestCPU(Spheres spheres,
    int len,
    float3 eye_point,
    float3 direction,
    float3* intersection,
    float* closest_distance,
    Mode mode,
    bool lightMode = false)
{
    int any = -1;
    // iteracja po kazdej sferze
    for (int i = 0; i < len; i++) {

        // pobranie wartosci paramtrow sfery
        float3 center = spheres.center[i];
        float radius = spheres.radius[i];

        float3 originCenter = eye_point - center;
        float a = dot(direction, direction);
        float b = 2.0f * dot(originCenter, direction);
        float c = dot(originCenter, originCenter) - radius * radius;
        // wyliczenie przeciecia z rownania kwadratowego
        float delta = b * b - 4 * a * c;
        float dist_current = -1;
        float t_local = -1;
        float3 intersection_current = { 0,0,0 };

        // brak przeciecia
        if (delta < 0) continue;
        // przeciecie w dwoch miejscach
        else if (delta > 0)
        {
            float t1 = (-b + sqrtf(delta)) / (2.0f * a);
            float t2 = (-b - sqrtf(delta)) / (2.0f * a);
            float tmin = min(t1, t2);
            float tmax = max(t1, t2);
            // bierzemy pierwszy wartosc wieksza niz 0 (w naszym przypadku 0.001)
            if (tmin > 0.001f)
            {
                intersection_current = eye_point + tmin * direction;
                t_local = tmin;
            }
            else if (tmin < 0.001f && tmax < 0.001f) continue;
            else
            {
                intersection_current = eye_point + tmax * direction;
                t_local = tmax;
            }

        }
        // przeciecie w jednym miejscu
        else
        {
            float t = -b / (2.0f * a);
            if (t < 0.001f) continue;
            intersection_current = eye_point + t * direction;
            t_local = t;
        }
        dist_current = length(intersection_current - eye_point);
        // wyznaczenie nowej najblizszej sfery 
        if (dist_current < *closest_distance || *closest_distance < 0.001)
        {
            *closest_distance = dist_current;
            *intersection = intersection_current;
            any = i;
            // sprawdzenie przeciecia dla cieni
            if (lightMode && t_local<1 && t_local>-1)
                return 1;
        }
    }
    return any;
}

////////////////////////////////////////////////////////////////////////////////
// Function that calculates pixel color
////////////////////////////////////////////////////////////////////////////////
__host__ void RayCastingSpheresCPU(
    uchar4* dst,
    const int imageW,
    const int imageH,
    Spheres spheres,
    const int spheres_len,
    Lights lights,
    const int lights_len,
    Screen2 screen,
    Mode mode,
    int x, 
    int y,
    bool shadow = false
)
{
    float3 eye_point = screen.eye_point;

    // wektor kierunku promienia
    float3 direction = screen.lower_left_corner
        + ((float)y / imageH) * screen.vertical
        + ((float)x / imageW) * screen.horizontal - eye_point;

    // indeks i odleglosc do najblizszej sfery (-1 - promien nie uderza w sfere)
    int i = -1;
    float closest_distance = -1;
    // punkt uderzenia promienia w sfere
    float3 intersection;

    // funkcja obliczajaca najblizsza sfere
    i = CalculateHitToClosestCPU(spheres, spheres_len,
        eye_point, direction,
        &intersection, &closest_distance,
        mode);

    // jesli promien nie uderzyl zadnej sfery
    if (i < 0 || closest_distance < 0.001f)
    {
        float3 interLight;
        float dist_light = -1;

        // wyliczenie i wizualizacja polozenia swiatla
        int light_i = CalculateSun(lights, lights_len, eye_point, direction, &interLight, &dist_light);
        if (light_i > 0)
        {
            dst[y * imageW + x] = { lights.color[light_i].x,lights.color[light_i].y,lights.color[light_i].z,0 };
            return;
        }

        // wyliczenia koloru tla
        float t = ((float)y) / imageH;
        float3 start = { 255.0,255.0,255.0 };
        float3 end = { 0.5 * 255.0,0.7 * 255.0,255.0 };
        float3 col = lerp(start, end, t);
        dst[y * imageW + x] = {
            static_cast<unsigned char>(col.x),
            static_cast<unsigned char>(col.y),
            static_cast<unsigned char>(col.z),
            0
        };
        return;
    }

    float3 sphere_center = spheres.center[i];
    float sphere_radius = spheres.radius[i];

    // wektor normalny do sfery w punkcie uderzenia promienia
    float3 normalVector = normalize(intersection - sphere_center);
    // wspolczynniki poczatkowe kolorow
    float3 color = { 0,0,0 };
    // kolor sfery
    float3 sphere_color = make_colorf(spheres.color[i]);

    // wyliczenia koloru dla kazdego swiatla jako skladowe modelu Phonga
    for (int j = 0; j < lights_len; j++)
    {
        float closest_distance_light = -1;
        float3 intersection_light;
        float3 direction_light = lights.position[j] - intersection;

        // wyliczenie cieni (opcjonalne)
        if (shadow)
        {
            // sprawdzenie czy zaden element nie rzuca cienia od danego swiatla na dany punkt sfery
            int inters = CalculateHitToClosestCPU(spheres, spheres_len,
                intersection, direction_light,
                &intersection_light, &closest_distance_light, mode, true);

            // jesli tak to nie dodajemy skladowych od danego swiatla
            if (inters > -1) continue;
        }

        // wyliczenie skladowych swiatla
        float3 light_color = make_colorf(lights.color[j]);
        direction_light = normalize(direction_light);
        float dotNL = clamp(dot(normalVector, direction_light), 0.0f, 1.0f);
        float3 r = normalize(2.0f * dotNL * direction_light - normalVector);
        float3 v = normalize(eye_point - intersection);

        float diffuse = clamp(dotNL, 0.0f, 1.0f);
        float specular = clamp(powf(fmaxf(dot(r, v), 0.0f), 64), 0.0f, 1.0f);
        color += (specular + diffuse) * light_color / 255.0f;
    }
    // ograniczenie kolorow w zakresie RGB (0-255)
    color = clamp(sphere_color * color, 0, 255);
    dst[y * imageW + x] = {
                static_cast<unsigned char>(color.x),
                static_cast<unsigned char>(color.y),
                static_cast<unsigned char>(color.z),
                0
    };
}

void rayCastingCPU(uchar4* dst,
        const int imageW,
        const int imageH,
        Spheres spheresD,
        int len,
        Lights lights_D,
        int len_lights,
        Camera camera,
        Mode mode)
{
        uchar4* h_dst = new uchar4[imageH * imageW];
        Screen2 screen = CalculateScreen(camera, imageW, imageH);
        switch (mode)
        {
        case CPURayCastingWithShadows:
            {
                for (int x = 0; x < imageW; x++)
                {
                    for (int y = 0; y < imageH; y++)
                    {
                        RayCastingSpheresCPU(h_dst, imageW, imageH, spheresD, len, lights_D, len_lights, screen, mode, x, y, true);
                    }
                }

                cudaMemcpy(dst, h_dst, sizeof(uchar4) * imageH * imageW, cudaMemcpyHostToDevice);
                delete[] h_dst;
                return;
            }
        case CPURayCasting:
            {
                for (int x = 0; x < imageW; x++)
                {
                    for (int y = 0; y < imageH; y++)
                    {
                        RayCastingSpheresCPU(h_dst, imageW, imageH, spheresD, len, lights_D, len_lights, screen, mode, x, y);
                    }
                }
                cudaMemcpy(dst, h_dst, sizeof(uchar4) * imageH * imageW, cudaMemcpyHostToDevice);
                delete[] h_dst;
                return;
            }
        }
    }

void rayCasting(uchar4* dst,
    const int imageW,
    const int imageH,
    SphereSoA spheresD,
    int len,
    Lights lights_D,
    int len_lights,
    Camera camera,
    Mode mode
) {

    const dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    const dim3 blocks(BLOCKS(imageW, BLOCKDIM_X), BLOCKS(imageH, BLOCKDIM_Y));
    Screen2 screen = CalculateScreen(camera, imageW, imageH);
    switch (mode)
        {
        case GPURayCastingWithShadowsConstantMemory:
        case GPURayCastingWithShadowsGlobalMemory:
            RayCastingSpheres << <blocks, threads >> > (dst, imageW, imageH, spheresD, len, lights_D, len_lights, screen, mode, true);
            return;
        case GPURayCastingConstantMemory:
        case GPURayCastingGlobalMemory:
            RayCastingSpheres << <blocks, threads >> > (dst, imageW, imageH, spheresD, len, lights_D, len_lights, screen, mode);
            return;
        }
}


void CopyToConstantData(Spheres spheres, int len)
{
    if (len > MAX_SPHERES_IN_CONSTANT_MEMORY) return;

    Sphere* tmp = new Sphere[len];
    for (int i = 0; i < len; i++)
    {
        float4 center = make_float4(spheres.center[i], spheres.radius[i]);
        tmp[i] = { center, spheres.color[i] };
    }
    cudaMemcpyToSymbol(spheres_constant, tmp, sizeof(Sphere) * len);

    delete[] tmp;
}
