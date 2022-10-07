#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"


__host__ __device__ float hashN(float n)
{
    return glm::fract(sin(n) * 43758.5453);
}

__host__ __device__ float random2D(glm::vec2 x)
{
    glm::vec2 p = floor(x);
    glm::vec2 f = fract(x);

    f = f * f * (3.0f - 2.0f * f);

    float n = p.x * 57.0 + p.y * 50.0;

    return glm::mix(glm::mix(hashN(n + 0.f), hashN(n + 1.f), f.x),
        glm::mix(hashN(n + 57.f), hashN(n + 58.f), f.x), f.y);
}

__host__ __device__ float interpNoise2D(glm::vec2 p) {
    int intX = int(floor(p.x));
    float fractX = glm::fract(p.x);
    int intY = int(floor(p.y));
    float fractY = glm::fract(p.y);

    float v1 = random2D(glm::vec2(intX, intY));
    float v2 = random2D(glm::vec2(intX + 1, intY));
    float v3 = random2D(glm::vec2(intX, intY + 1));
    float v4 = random2D(glm::vec2(intX + 1, intY + 1));

    float i1 = glm::mix(v1, v2, fractX);
    float i2 = glm::mix(v3, v4, fractX);
    return glm::mix(i1, i2, fractY);
}

__host__ __device__ float fbm2D(glm::vec2 p, float freq, float persistence, float amp) {
    float total = 0.f;
    int octaves = 8;
    for (int i = 0; i < octaves; i++) {
        freq *= pow(2.0f, float(i));
        amp *= pow(persistence, float(i));
        total += interpNoise2D(p * freq) * amp;
    }
    return total;
}

__host__ __device__ float random3D(glm::vec3 x)
{
    glm::vec3 p = floor(x);
    glm::vec3 f = fract(x);

    f = f * f * (3.0f - 2.0f * f);

    float n = p.x + p.y * 57.0 + p.z * 50.0;

    return glm::mix(glm::mix(hashN(n + 0.0), hashN(n + 1.0), f.x),
        glm::mix(hashN(n + 57.0), hashN(n + 58.0), f.x), f.y);
}


__host__ __device__ float interpNoise3D(glm::vec3 p) {
    int intX = int(floor(p.x));
    float fractX = glm::fract(p.x);
    int intY = int(floor(p.y));
    float fractY = glm::fract(p.y);
    int intZ = int(floor(p.z));
    float fractZ = glm::fract(p.z);

    float v1 = random3D(glm::vec3(intX, intY, intZ));
    float v2 = random3D(glm::vec3(intX, intY, intZ + 1));
    float v3 = random3D(glm::vec3(intX, intY + 1, intZ));
    float v4 = random3D(glm::vec3(intX, intY + 1, intZ + 1));

    float v5 = random3D(glm::vec3(intX + 1, intY, intZ));
    float v6 = random3D(glm::vec3(intX + 1, intY, intZ + 1));
    float v7 = random3D(glm::vec3(intX + 1, intY + 1, intZ));
    float v8 = random3D(glm::vec3(intX + 1, intY + 1, intZ + 1));

    float i1 = glm::mix(v1, v2, fractZ);
    float i2 = glm::mix(v3, v4, fractZ);
    float i3 = glm::mix(i1, i2, fractY);

    float i4 = glm::mix(v5, v6, fractZ);
    float i5 = glm::mix(v7, v8, fractZ);
    float i6 = glm::mix(i4, i5, fractY);

    return glm::mix(i3, i6, fractX);
}

__host__ __device__ float fbm3D(glm::vec3 p, float freq, float persistence) {
    float total = 0.f;
    //float persistence = 0.5f;
    int octaves = 8;
    //float freq = 2.f;
    float amp = 0.5f;
    for (int i = 1; i <= octaves; i++) {
        total += interpNoise3D(p * freq) * amp;
        freq *= 2.f;
        amp *= persistence;
    }
    return total;
}
 __host__ __device__ float smoothstep(float a, float b, float x)
 {
     float X = ((x - a) / (b - a));
     float t = max(0.f, min(1.f, X));
     return t * t * (3.0 - (2.0 * t));
 }

__host__ __device__ glm::vec3 getProceduralColor(PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal) {

    glm::vec3 isectCpy = glm::normalize(intersect) * 2.f - 1.f;
    glm::vec4 baseCol = glm::vec4(1.0, 1.0, 0.0, 1.0);
    float theta = glm::atan(intersect.x, intersect.y);
    float r = sqrt(pow(intersect.x, 2.0f) + pow(intersect.y, 2.0f));
    glm::vec4 brown = glm::vec4(0.36, 0.29, 0.01, 1.0) * fbm3D(glm::vec3(isectCpy), 2.f, 0.5f);
    glm::vec4 green = glm::vec4(0.24, abs(sin(0.4 * r * 20.f)), cos(0.22 * r * 90.f), 1.0);
    glm::vec4 blue = glm::vec4(0.0, 0.0, 1.0, 1.0);

    glm::vec4 out_color = glm::vec4(0.0, 0.0, 0.0, 1.0);
   
    float smoothVal = smoothstep(0.9, 0.99f, isectCpy.z);
   
     // blue with low persistence noise function
    float f_blue = fbm3D(glm::vec3(intersect), 10.f, 0.2f);
    blue *= f_blue;

    // increasing blue influence with radius and adjusting green ring position with appropriate multiple
    float f_green = fbm3D(glm::vec3(intersect), 20.f, 0.5f);
    green *= f_green;

    // multiplying with cos(theta) makes cos wave look like rays around the center
    // use sin function based on z to set a varying phase difference
    //float f_brown = fbm3D(glm::vec3(intersect), 2.f, 0.5f);
    //brown = glm::vec4(0.36, 0.29, 0.01, 1.0) * f_brown;
    //brown *= abs(cos(theta * 20.f - sin(isectCpy.z * interpNoise3D(intersect) * 200.f)));// *3.14f / 8.f));

    // In smoothstep function below, fs_Pos.z value changes from 0.98 to 0.99, value gradually decreases
    // Subtracting it from 1.0 inverses the effect, so as fs_Pos.z moves towards 0.99, the value increases
    smoothVal = smoothstep(0.98, 0.99f, isectCpy.z);

    out_color = mix(green, blue, 0.5f);// mix(mix(green, blue, 0.5f), brown, f_brown);
    out_color *= (1.f - smoothVal);

    return glm::vec3(out_color.x, out_color.y, out_color.z);
}