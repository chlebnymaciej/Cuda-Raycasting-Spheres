#pragma once
#include "RayCPU.h"
#include <helper_math.h>
#include <stdio.h>


////////////////////////////////////////////////////////////////////////////////
// Function that calculates if rays hits light directly
////////////////////////////////////////////////////////////////////////////////
int CalculateSun(Lights lights, int len, float3 eye_point, float3 direction,
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
        float t_local = -1;
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
                t_local = tmin;
            }
            else if (tmin < 0.001f && tmax < 0.001f) continue;
            else
            {
                intersection_current = eye_point + tmax * direction;
                t_local = tmax;
            }

        }
        else
        {
            float t = -b / (2.0f * a);
            if (t < 0.001f) continue;
            intersection_current = eye_point + t * direction;
            t_local = t;
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
int CalculateHitToClosest(Spheres spheres,
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
void RayCastingSpheresCPU(
    uchar4* dst,
    const int imageW,
    const int imageH,
    Spheres spheres,
    const int spheres_len,
    Lights lights,
    const int lights_len,
    Screen screen,
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
    i = CalculateHitToClosest(spheres, spheres_len,
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
        dst[y * imageW + x] = { (unsigned char)col.x,(unsigned char)col.y,(unsigned char)col.z,0 };
        return;
    }

    // wektor normalny do sfery w punkcie uderzenia promienia
    float3 normalVector = normalize(intersection - spheres.center[i]);
    // wspolczynniki poczatkowe kolorow
    float3 color = { 0,0,0 };
    // kolor sfery
    float3 sphere_color = {spheres.color[i].x,spheres.color[i].y,spheres.color[i].z};


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
        float3 light_color = { lights.color[j].x, lights.color[j].y,lights.color[j].z };
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
    dst[y * imageW + x] = { (unsigned char)color.x, (unsigned char)color.y, (unsigned char)color.z, 0 };
}
////////////////////////////////////////////////////////////////////////////////
// Function that calculates window coordinates
////////////////////////////////////////////////////////////////////////////////
Screen CalculateScreen(Camera cam, int imageW, int imageH)
{
    // Obliczamy wycinek plaszczyzny odpowiadaj¹cy naszemu oknu


    const float x = cam.look_from.x + cam.r * cosf(cam.theta * radianCalc) * cosf(cam.fi * radianCalc);
    const float z = cam.look_from.z + cam.r * cosf(cam.theta * radianCalc) * sinf(cam.fi * radianCalc);
    const float y = cam.look_from.y + cam.r * sinf(cam.theta * radianCalc);
    float3 look_at = { x,y,z };


    // kat jaki widzimy w pionie od srodka do gronej krawedzi
    float theta = cam.vfov;
    float h = tanf(theta / 2);
    // przeskalowana wysokosc
    float viewport_height = 2.0 * h;
    // przeskalowana szerosc z zachowaniem proporcji ekranu
    float viewport_width = (imageW * viewport_height) / imageH;

    // wirtualny punkt z ktorego wychodza promienie
    float3 eye_point = cam.look_from;

    // przesuniecie plaszczyzny po osiach
    float3 z_wersor = normalize(cam.look_from - look_at);
    float3 x_wersor = normalize(cross(cam.vup, z_wersor));
    float3 y_wersor = cross(z_wersor, x_wersor);

    float3 horizontal = viewport_width * x_wersor;
    float3 vertical = viewport_height * y_wersor;
    float3 lower_left_corner = eye_point - horizontal / 2 - vertical / 2 - z_wersor;

    Screen screen = { eye_point, horizontal, vertical, lower_left_corner };
    return screen;
}


void rayCasting(uchar4* dst,
    const int imageW,
    const int imageH,
    Spheres spheresD,
    int len,
    Lights lights_D,
    int len_lights,
    Camera camera,
    Mode mode
) {


    Screen screen = CalculateScreen(camera, imageW, imageH);
    for (int x = 0; x < imageW; x++)
    {
        for (int y = 0; y < imageH; y++)
        {
            RayCastingSpheresCPU(dst, imageW, imageH, spheresD, len, lights_D, len_lights, screen, mode,x,y);
        }
    }
}