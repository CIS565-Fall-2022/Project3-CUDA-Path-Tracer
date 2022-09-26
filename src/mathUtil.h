#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define Pi 3.1415926535897932384626422832795028841971f
#define PiTwo 6.2831853071795864769252867665590057683943f
#define PiInv 1.f / Pi
#define OneThirdSqrt 0.5773502691896257645091487805019574556476f
#define EpsCmp 0.00001f

namespace Math {
    bool epsilonCheck(float a, float b);
    glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);

    __host__ __device__ inline float satDot(glm::vec3 a, glm::vec3 b) {
        return glm::max(glm::dot(a, b), 0.f);
    }

    __host__ __device__ inline float absDot(glm::vec3 a, glm::vec3 b) {
        return glm::abs(glm::dot(a, b));
    }

    __host__ __device__ inline float pow5(float x) {
        float x2 = x * x;
        return x2 * x2 * x;
    }

    __host__ __device__ inline float square(float x) {
        return x * x;
    }

    template<typename T>
    __host__ __device__ inline T calcFilmic(T c) {
        return (c * (c * 0.22f + 0.03f) + 0.002f) / (c * (c * 0.22f + 0.3f) + 0.06f) - 1.f / 30.f;
    }

    __host__ __device__ inline glm::vec3 filmic(glm::vec3 c) {
        return calcFilmic(c * 1.6f) / calcFilmic(11.2f);
    }

    __host__ __device__ inline glm::vec3 ACES(glm::vec3 color) {
        return (color * (color * 2.51f + 0.03f)) / (color * (color * 2.43f + 0.59f) + 0.14f);
    }

    __host__ __device__ inline glm::vec3 correctGamma(glm::vec3 color) {
        return glm::pow(color, glm::vec3(1.f / 2.2f));
    }

    /**
    * Map a pair of evenly distributed [0, 1] coordinate to disc
    */
    __device__ static glm::vec2 toConcentricDisk(float x, float y) {
        float r = glm::sqrt(x);
        float theta = y * Pi * 2.0f;
        return glm::vec2(glm::cos(theta), glm::sin(theta)) * r;
    }

    __device__ static glm::mat3 localRefMatrix(glm::vec3 n) {
        glm::vec3 t = (glm::abs(n.y) > 0.9999f) ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(0.f, 1.f, 0.f);
        glm::vec3 b = glm::normalize(glm::cross(n, t));
        t = glm::cross(b, n);
        return glm::mat3(t, b, n);
    }

    __device__ static glm::vec3 localToWorld(glm::vec3 n, glm::vec3 v) {
        return glm::normalize(localRefMatrix(n) * v);
    }

    __device__ static glm::vec3 sampleHemisphereCosine(glm::vec3 n, float rx, float ry) {
        glm::vec2 d = toConcentricDisk(rx, ry);
        float z = glm::sqrt(1.f - glm::dot(d, d));
        return localToWorld(n, glm::vec3(d, z));
    }

    __device__ static bool refract(glm::vec3 n, glm::vec3 wi, float ior, glm::vec3& wt) {
        float cosIn = glm::dot(n, wi);
        if (cosIn < 0) {
            ior = 1.f / ior;
        }
        float sin2In = glm::max(0.f, 1.f - cosIn * cosIn);
        float sin2Tr = sin2In / (ior * ior);

        if (sin2Tr >= 1.f) {
            return false;
        }
        float cosTr = glm::sqrt(1.f - sin2Tr);
        if (cosIn < 0) {
            cosTr = -cosTr;
        }
        wt = glm::normalize(-wi / ior + n * (cosIn / ior - cosTr));
        return true;
    }

    __device__ inline float areaPdfToSolidAngle(float pdf, glm::vec3 ref, glm::vec3 y, glm::vec3 ny) {
        glm::vec3 yToRef = ref - y;
        return pdf * absDot(ny, glm::normalize(yToRef)) / glm::dot(yToRef, yToRef);
    }

    template<typename T>
    size_t byteSizeOfVector(const std::vector<T>& v) {
        return v.size() * sizeof(T);
    }

    /**
    * Handy-dandy hash function that provides seeds for random number generation.
    */
    __host__ __device__ inline unsigned int utilhash(unsigned int a) {
        a = (a + 0x7ed55d16) + (a << 12);
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        a = (a + 0x165667b1) + (a << 5);
        a = (a + 0xd3a2646c) ^ (a << 9);
        a = (a + 0xfd7046c5) + (a << 3);
        a = (a ^ 0xb55a4f09) ^ (a >> 16);
        return a;
    }
}