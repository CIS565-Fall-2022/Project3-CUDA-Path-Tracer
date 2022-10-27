#pragma once

__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine& rng);

__host__ __device__
void sampleLight(
	thrust::default_random_engine& rng,
    const Geom& light,
    glm::vec3& newDir,
    const glm::vec3& intersect,
    float& pdf_light,
    const int num_lights
) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    if (light.type == SQUARE_PLANE) {
        float x = u01(rng) - 0.5f;
        float z = u01(rng) - 0.5f;
        glm::vec3 sampleInv(x, 0.f, z);
        glm::vec3 sample = multiplyMV(light.transform, glm::vec4(sampleInv, 1));
        float lightArea = light.scale.x * light.scale.z;
        float pdf_dA = 1.f / lightArea;
        newDir = sample - intersect;
        float r_square = glm::length2(newDir);
        //can't directly use light.rotation, since light.rotation is all 0
        glm::vec3 lightNormal = glm::normalize(glm::mat3(light.transform) * glm::vec3(0, 1, 0));
        //calculate cos_theta
        newDir = glm::normalize(newDir);
        float cos_theta = glm::abs(glm::dot(newDir, lightNormal));
        pdf_light = (pdf_dA * r_square / cos_theta) / num_lights;
    }
}

__host__ __device__
void sampleF(
    thrust::default_random_engine& rng,
    const Material& m,
    glm::vec3& newDir,
    const glm::vec3& intersect,
    float& pdf_f,
    const glm::vec3& normal,
    const PathSegment& pathSegment
) {
    if (m.hasReflective == 1.f) {
        //mirror
        //glm::reflect  (1,1,0) (0, 1, 0) => (1, -1, 0)
        newDir = glm::reflect(pathSegment.ray.direction, normal);
        pdf_f = 1.f;
    }
    else if (m.hasRefractive == 1.f) {
        //refract

    }
    else {
        //lambert
        newDir = calculateRandomDirectionInHemisphere(normal, rng);
        pdf_f = glm::dot(newDir, normal) * INV_PI;
        
    }
}