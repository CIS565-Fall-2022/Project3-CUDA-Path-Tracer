#pragma once

// #define NO_DEFAULT_PATHS


#define PRIM_SPHERE_RADIUS 0.5f
#define PRIM_CUBE_EXTENT 0.5f
#define BACKGROUND_COLOR (glm::vec3(0.0f))
#define NUM_TEX_CHANNEL 4
#define MAX_EMITTANCE 100.0f
#define WORLD_UP (glm::vec3(0, 1, 0))

#define BLOCK_SIZE 128

// maximum number of kernels launched for intersection test
// #define MAX_INTERSECTION_TEST_SIZE BLOCK_SIZE
#define LARGE_FLOAT (float)(1e10)
#define SMALL_FLOAT (float)(-1e10)
#define OCTREE_BOX_EPS 0.001f
#define OCTREE_DEPTH 3
#define OCTREE_MESH_ONLY

// impl switches
#define COMPACTION
// #define SORT_MAT
#define AABB_CULLING
#define OCTREE_CULLING
// #define DEPTH_OF_FIELD

// #define ANTI_ALIAS_JITTER
// #define FAKE_SHADE

#define PROFILE




#define DENOISE
#define DENOISE_USE_DIFFUSE_MAP
#define DENOISE_GBUF_OPTIMIZATION
// #define DENOISE_SHARED_MEM


// #define CACHE_FIRST_BOUNCE
#if (defined(CACHE_FIRST_BOUNCE) && defined(ANTI_ALIAS_JITTER)) || (defined(CACHE_FIRST_BOUNCE) && defined(DEPTH_OF_FIELD)) 
#error "ANTI_ALIAS_JITTER or CACHE_FIRST_BOUNCE cannot be used with CACHE_FIRST_BOUNCE"
#endif