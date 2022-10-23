#pragma once

#define DEFAULT_SKY_COLOR 0.1

// Skybox setting
#define ENABLE_SKYBOX 1

// Path trace setting
#define CACHE_FIRST_INTERSECTIONS 1
#define SORT_MATERIALS 0

// Anti-Aliasing
#define ENABLE_ANTI_ALIASING 0

// BVH setting
#define ENABLE_BVH  0
#define SAH_BUCKET_SIZE 12
#define MAX_PRIM_IN_BVH_NODE 100
#define BVH_INTERSECT_STACK_SIZE 128

// Texture setting
#define ENABLE_TEXTURE 1
#define ENABLE_NORMAL_MAP 1

// Denoiser
#define ENABLE_NOISER 1