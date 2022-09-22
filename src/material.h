#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "mathUtil.h"

#define InvalidPdf -1.f

struct Material {
    enum Type {
        Lambertian = 0, MetallicWorkflow = 1, Dielectric = 2, Light = 3
    };

    int type;
    glm::vec3 baseColor;
    float metallic;
    float roughness;
    float ior;
    float emittance;
};

enum BSDFSampleType {
    Diffuse = 1 << 0,
    Glossy = 1 << 1,
    Specular = 1 << 2,

    Reflection = 1 << 4,
    Transmission = 1 << 5
};

struct BSDFSample {
    glm::vec3 wi;
    glm::vec3 bsdf;
    float pdf;
    int type;
};

__device__ inline float fresnelApprox(float cosTheta) {
    return Math::pow5(1.f - cosTheta);
}

__device__ static glm::vec3 fresnelShlick(float cosTheta, glm::vec3 f0) {
    return glm::mix(f0, glm::vec3(1.f), Math::pow5(1.f - cosTheta));
}

__device__ static float fresnel(float cosIn, float ior) {
    if (cosIn < 0) {
        ior = 1.f / ior;
        cosIn = -cosIn;
    }
    float sinIn = glm::sqrt(1.f - cosIn * cosIn);
    float sinTr = sinIn / ior;
    if (sinTr >= 1.f) {
        return 1.f;
    }

    float cosTr = glm::sqrt(1.f - sinTr * sinTr);
    return (Math::square((cosIn - ior * cosTr) / (cosIn + ior * cosTr)) +
        Math::square((ior * cosIn - cosTr) / (ior * cosIn + cosTr))) * .5f;
}

__device__ static glm::vec3 lambertianBSDF(glm::vec3 n, glm::vec3 wo, glm::vec3 wi, const Material &m) {
    return m.baseColor * Math::satDot(n, wi) * PiInv;
}

__device__ static float lambertianPdf(glm::vec3 n, glm::vec3 wo, glm::vec3 wi, const Material& m) {
    return glm::dot(n, wi) * PiInv;
}

__device__ static void lambertianSample(glm::vec3 n, glm::vec3 wo, const Material& m, glm::vec3 r, BSDFSample& sample) {
    sample.wi = Math::sampleHemisphereCosine(n, r.x, r.y);
    sample.bsdf = m.baseColor * PiInv;
    sample.pdf = glm::dot(n, sample.wi) * PiInv;
    sample.type = Diffuse | Reflection;
}

__device__ static glm::vec3 dielectricBSDF(glm::vec3 n, glm::vec3 wi, const Material& m) {
    return glm::vec3(0.f);
}

__device__ static float dielectricPdf(glm::vec3 n, glm::vec3 wo, glm::vec3 wi, const Material& m) {
    return 0.f;
}

__device__ static void dielectricSample(glm::vec3 n, glm::vec3 wo, const Material& m, glm::vec3 r, BSDFSample& sample) {
    float ior = m.ior;
    float pdfRefl = fresnel(glm::dot(n, wo), ior);
    float pdfTran = 1.f - pdfRefl;

    sample.pdf = 1.f;
    sample.bsdf = m.baseColor;

    if (r.z < pdfRefl) {
        sample.wi = glm::reflect(-wo, n);
        sample.type = Specular | Reflection;
    }
    else {
        if (!Math::refract(n, wo, ior, sample.wi)) {
            sample.pdf = InvalidPdf;
            return;
        }
        if (glm::dot(n, wo) < 0) {
            ior = 1.f / ior;
        }
        sample.bsdf /= ior * ior;
        sample.type = Specular | Transmission;
    }
} 