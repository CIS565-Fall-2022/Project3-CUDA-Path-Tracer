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

__host__ __device__
void calculateLightPdf(
    const PathSegment& pathSegment,
    glm::vec3& tmp_intersect,
    glm::vec3& tmp_normal,
    bool& outside, 
    float& pdf_f_l,
    Geom& light
) {
    if (light.type == SQUARE_PLANE) {
        float t = squarePlaneIntersectionTest(light, pathSegment.ray, tmp_intersect, tmp_normal, outside);
    }
}