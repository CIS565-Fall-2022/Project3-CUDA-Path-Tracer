#pragma once

__host__ __device__
void calculateBSDF(
	const Material& m,
	glm::vec3& bsdf,
	float& bsdf_pdf,
    const glm::vec3& newDir,
    const PathSegment& pathSegment,
    const glm::vec3& normal,
    const glm::vec3& intersect
){
    if (m.hasReflective == 1.f) {
        if (newDir == glm::reflect(pathSegment.ray.direction, normal)) {
            bsdf = m.specular.color;
            bsdf_pdf = 1.f;
        }
        else {
            bsdf = glm::vec3(0.f);
            bsdf_pdf = 0.f;
        }
    }
    else if (m.hasRefractive == 1.f) {

    }
    else {
        //lambert
        glm::vec3 brdf_L = m.color * INV_PI;
        float lambertFactor = glm::dot(newDir, normal);
        bsdf = lambertFactor * brdf_L;
        bsdf_pdf = lambertFactor * INV_PI;
    }
}

/**
* 
* assume we hit the light, return its pdf
*/
__host__ __device__
void calculateLightPdf(
    const PathSegment& pathSegment,
    glm::vec3& orig_intersect,
    glm::vec3& tmp_intersect,
    glm::vec3& tmp_normal,
    bool& outside, 
    float& pdf_f_l,
    Geom& light,
    int num_lights
) {
    if (light.type == SQUARE_PLANE) {
        float lightArea = light.scale.x * light.scale.z;
        float pdf_dA = 1.f / lightArea;
        glm::vec3 newDir = tmp_intersect - orig_intersect;
        float r_square = glm::length2(newDir);
        glm::vec3 lightNormal = glm::normalize(glm::mat3(light.transform) * glm::vec3(0, 1, 0));
        newDir = glm::normalize(newDir);
        float cos_theta = glm::abs(glm::dot(newDir, lightNormal));
        if (cos_theta == 0.f) {
            pdf_f_l = 0.f;
        }
        else {
            pdf_f_l = (pdf_dA * r_square / cos_theta) / num_lights;
        }
    }
}