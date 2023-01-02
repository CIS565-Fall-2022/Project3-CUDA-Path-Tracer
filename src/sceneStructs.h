#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))
#define UNDEFINED_VEC4 (glm::vec4(0.f))

#define BVH 0
#define ROUGHNESS_METALLIC 1
#define SORT_BY_MATERIALS 1
// turn on at most ONE of first bounce caching and anti-aliasing
#define CACHE_FIRST_BOUNCE 0
#define ANTI_ALIAS 1
// for debugging
#define SHOW_NORMALS 0
#define SHOW_METALLIC 0

namespace scene_structs {

enum GeomType {
  SPHERE,
  CUBE,
  TRIANGLE_MESH,
};

struct Ray {
  glm::vec3 origin;
  glm::vec3 direction;
};

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec4 tangent = UNDEFINED_VEC4; // many meshes don't have tangent maps
  glm::vec2 uv;
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

  int triangleOffset; // for triangle_mesh type to index into triangle array
  int numTriangles;
  int bvhOffset; // each triangle mesh has its own bvh
};

struct Triangle {
  Vertex verts[3];
};

// intermediate struct to parse triangles out of
struct InputMesh {
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec2> uvCoords;
  std::vector<glm::vec4> tangents;
  std::vector<unsigned int> indices;
};

struct Image {
  int height;
  int width;
  std::vector<glm::vec3> pixels;
};

struct Material {
  int colorImageId = -1; // -1 means there is no texture
  int normalMapImageId = -1;
  int roughnessMetallicImageId = -1;
  float metallicFactor = 0;
  glm::vec3 color;
  struct {
    float exponent;
    glm::vec3 color;
  } specular;
  float hasReflective = 0;
  float hasRefractive = 0;
  float indexOfRefraction;
  float emittance = 0;
};

struct Camera {
  glm::ivec2 resolution; // eg. 1080 x 1920
  glm::vec3 position;
  glm::vec3 lookAt; // NOT necessarily normalized
  glm::vec3 view; // normalized, direction camera is facing
  glm::vec3 up; // normalized
  glm::vec3 right; // normalized
  glm::vec2 fov;
  glm::vec2 pixelLength;
};

struct RenderState {
  Camera camera;
  unsigned int iterations;
  int traceDepth; // DEPTH value in scene txt file
  std::vector<glm::vec3> image;
  std::string imageName;
};

// a path stores the multiple ray bounces from one pixel until the ray dies
// GUESSING HERE: pixel index is where the light came from (backwards from the camera)
// Ray would be... the current? ray that the path is on
// color would be... the amount of color that's been ACCUMULATED? from this path
// remainingBounces self explanatory
// We have 1 pathsegment per pixel?
// Guess we only need 1 pathsegment to consolidate multiple paths per pixel (for anti-aliasing) 
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
  glm::vec3 surfaceNormal; // normalized
  glm::vec4 surfaceTangent; // to convert tangent space normal from texture into normal
  glm::vec2 uv;
  int materialId;
};

}