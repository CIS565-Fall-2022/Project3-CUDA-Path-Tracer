#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
	SPHERE,
	CUBE,
	MESH
};

struct Triangle {
	glm::vec3 pos[3];
	glm::vec3 normal[3];
	glm::vec2 uv[3];
	glm::vec4 tangent[3];
};

struct Ray {
	glm::vec3 origin;
	glm::vec3 direction;
};

struct Map {
	int offset = -1;
	glm::ivec2 dim;
};

struct Geom {
	enum GeomType type;
	int materialid;
	glm::vec3 translation;
	glm::vec3 rotation;
	glm::vec3 scale;
	glm::mat4 transform;
	glm::mat4 inverseTransform;
	glm::mat4 invTranspose;
	int mesh_start_idx;
	int mesh_end_idx;
};

struct Material {
	glm::vec3 color;
	struct {
		float exponent;
		glm::vec3 color;
	} specular;
	float hasReflective;
	float hasRefractive;
	float indexOfRefraction;
	float emittance;
	Map texture_map;
	Map normal_map;
};

struct Camera {
	glm::ivec2 resolution;
	glm::vec3 position;
	glm::vec3 lookAt;
	glm::vec3 view;
	glm::vec3 up;
	glm::vec3 right;
	glm::vec2 fov;
	glm::vec2 pixelLength;
	float focalLength;
	float aperture;
};

struct RenderState {
	Camera camera;
	unsigned int iterations;
	int traceDepth;
	std::vector<glm::vec3> image;
	std::string imageName;
};

struct PathSegment {
	Ray ray;
	glm::vec3 color;
	int pixelIndex;
	int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
	float t;
	glm::vec3 surfaceNormal;
	int materialId;
	glm::vec2 uv;
};
