#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include <iostream>
#include <device_functions.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define CACHE_INTERSECTION 1
#define SORT_RAY 1
#define ANTI_ALIASING 1
#define DOF 0
#define DIRECTLIGHTING 0
#define POSTPROCESS 0
#define RED 0
#define GREEN 0
#define BLUE 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

//Predicate functor
struct is_not_zero {
    __host__ __device__
        bool operator()(const PathSegment& x) {
        return x.remainingBounces;
    }
};

struct compareMatId {
    __host__ __device__
        bool operator()(const ShadeableIntersection& left, const ShadeableIntersection& right) {
        return left.materialId < right.materialId;
    }
};

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;

static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static ShadeableIntersection* dev_cache_intersections = NULL;//for first round cache
static Primitive* dev_primitives = NULL;//for mesh
static Texture* dev_textures = NULL;
static glm::vec3* dev_texData = nullptr;
//for more detailed GLTF(loading multiple textures without actually referening them in txt file
static glm::vec3* dev_lightPoint = nullptr;
static cudaTextureObject_t* dev_cudaTexObjs = NULL;
static std::vector<cudaArray_t> dev_arrays;
static std::vector<cudaTextureObject_t> dev_texs;

//learned from GPU Gem
__host__ void textureInitGPU(Texture& tex, int i) {
    cudaTextureObject_t texObj;
    int width = tex.width;
    int height = tex.height;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    cudaMallocArray(&dev_arrays[i], &desc, width, height);
    cudaMemcpyToArray(dev_arrays[i], 0, 0, tex.image, width * height * tex.components * sizeof(unsigned char), cudaMemcpyHostToDevice);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = dev_arrays[i];

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.sRGB = 1;
    texDesc.normalizedCoords = 1;

    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    cudaMemcpy(dev_cudaTexObjs + i, &texObj, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    checkCUDAError("textureInit failed");

    dev_texs.push_back(texObj);
}

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_cache_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_primitives, scene->primitives.size() * sizeof(Primitive));
    cudaMemcpy(dev_primitives, scene->primitives.data(), scene->primitives.size() * sizeof(Primitive), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
    cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);
    //I use two different method for texture selecting, one is gltf auto loading(referenced from GPU Gem CudaTextureObj Part) and one is custom selected;
    if (scene->texData.size() > 0)
    {
        cudaMalloc(&dev_texData, scene->texData.size() * sizeof(glm::vec3));
        cudaMemcpy(dev_texData, scene->texData.data(), scene->texData.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    }
    //auto gltf loading
    //create texture memory
    dev_arrays.clear(); dev_texs.clear();
    cudaMalloc(&dev_cudaTexObjs, scene->textures.size() * sizeof(cudaTextureObject_t));
    dev_arrays.resize(scene->textures.size());
    for (int i = 0; i < scene->textures.size(); i++) {
        textureInitGPU(scene->textures[i], i);
    }

    cudaMalloc(&dev_lightPoint, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_lightPoint, 0, pixelcount * sizeof(glm::vec3));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    
    // TODO: clean up any extra device memory you created
    cudaFree(dev_cache_intersections);
    cudaFree(dev_primitives);
    cudaFree(dev_textures);
    cudaFree(dev_lightPoint);

    for (int i = 0; i < dev_texs.size(); i++) {
        cudaFreeArray(dev_arrays[i]);
        cudaDestroyTextureObject(dev_texs[i]);
    }
    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment & segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        
#if ANTI_ALIASING//Stochastic Anti Aliasing Implementation
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
        );
#else 
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
#endif


#if DOF//Depth of Field Implementation
        cam.focal_length = 10.0f;
        cam.aperture_radius = 0.2f;

        //float ft = glm::abs((cam.focal_length) / segment.ray.direction.z);
        glm::vec3 focalPoint = segment.ray.direction * cam.focal_length;
        //two steps: shift ray.origin and change ray.direction

        glm::vec3 shiftIdx = glm::vec3(u01(rng) - 0.5f, u01(rng) - 0.5f, 0.f);
        shiftIdx *= cam.aperture_radius;
        segment.ray.origin += shiftIdx;
        segment.ray.direction = glm::normalize(focalPoint - glm::vec3(shiftIdx.x, shiftIdx.y, 0.f));

#endif // DOF

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth
    , int num_paths
    , PathSegment * pathSegments
    , Geom * geoms
    , int geoms_size
    , ShadeableIntersection * intersections
    , Primitive* prims
    , Material* material
    , glm::vec3* texData
    )
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::vec4 tangent;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;
        int tmp_matId = -1;
        int matId = -1;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;
        glm::vec4 tmp_tangent;

        // naive parse through global geoms
        for (int i = 0; i < geoms_size; i++)
        {
            Geom & geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                tmp_matId = geom.materialid;
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                tmp_matId = geom.materialid;
            }
            else if (geom.type == MESH) {
                t = primitiveIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, tmp_tangent, prims, material, texData);
                tmp_matId = geom.matId;
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                matId = tmp_matId;
                uv = tmp_uv;
                tangent = tmp_tangent;
            }
        }

        ShadeableIntersection& intersection = intersections[path_index];
        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            //The ray hits something
            intersection.t = t_min;
            intersection.materialId = matId;
            intersection.surfaceNormal = normal;
            intersection.uv = uv;
            intersection.tangent = tangent;
        }
    }
}

//ShadeDirectLight: Do a per-pixel light scan to get every dev_lightPoint
__global__ void shadeDirectLight(int iter, int num_paths, ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments
    , Material* materials, Texture* textures, glm::vec3* dev_texData, glm::vec3* dev_lightPoint) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) {
            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;
            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                dev_lightPoint[idx] = getPointOnRay(pathSegments[idx].ray, intersection.t);
            }
        }
    }
    //__syncthreads();
}

//findClosestLight tries to find either the light position at the point or the earliest light position in the screen coordinates
__device__ glm::vec3 findClosestLight(int idx, int numPath, glm::vec3* dev_lightPoint) {
    if (dev_lightPoint[idx] != glm::vec3(0, 0, 0)) {
        return dev_lightPoint[idx];
    }
    for (int i = 0; i < numPath; i++) {
        if (dev_lightPoint[i] != glm::vec3(0, 0, 0)) {
            return dev_lightPoint[i];
        }
    }
    return glm::vec3(0.f, 0.f, 0.f);
}

// LOOK: shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeMaterial(int iter, int num_paths, ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments
    , Material* materials, Texture* textures, glm::vec3* dev_texData, glm::vec3* dev_lightPoint, cudaTextureObject_t* cudaTex) {

    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment& pathSegment = pathSegments[idx];
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { 
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.gltf ? material.pbrMetallicRoughness.baseColorFactor : material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f || material.emissiveTexture.index >= 0) {
                if (material.gltf) {
                    glm::vec3 EmissiveColor = material.emissiveFactor * sample(cudaTex[material.emissiveTexture.index], intersection.uv);
                    if (glm::length(EmissiveColor) > 0.0f) {
                        pathSegment.color *= EmissiveColor;
                        pathSegment.remainingBounces = 0;
                    }
                    else {
                        scatterRayGLTF(pathSegment, getPointOnRay(pathSegment.ray, intersection.t), intersection.surfaceNormal, intersection.uv, intersection.tangent, material, rng, cudaTex);
                        pathSegment.remainingBounces--;
                    }
                }
                else {
                    pathSegment.color *= (materialColor * material.emittance);
                    pathSegment.remainingBounces = 0;
                }
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            else {
                if (material.gltf) {
                    scatterRayGLTF(pathSegment, getPointOnRay(pathSegment.ray, intersection.t), intersection.surfaceNormal, intersection.uv, intersection.tangent, material, rng, cudaTex);
                }
                else {
                    scatterRay(pathSegment, getPointOnRay(pathSegment.ray, intersection.t), intersection.surfaceNormal, intersection.uv, material, rng, textures, dev_texData, cudaTex);
                }
                if (DIRECTLIGHTING && pathSegment.remainingBounces == 2) {
                    glm::vec3 lightPoint = findClosestLight(idx, num_paths, dev_lightPoint);
                    if (lightPoint != glm::vec3(0.f, 0.f, 0.f)) {
                        pathSegment.ray.direction = glm::normalize(lightPoint - pathSegment.ray.origin);
                    }//just change the direction so next ray it either cast to object or the light(very basic so we dont worry about the former)
                }
                pathSegment.remainingBounces--;
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

//basic Post Process Color Tinting process
__global__ void shadePostProcess(int iter, int num_paths, ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        glm::vec3 postProcessColor;
#if RED
        postProcessColor = glm::vec3(1.f, 0.5f, 0.5f);
#endif
#if GREEN
        postProcessColor = glm::vec3(0.5f, 1.f, 0.5f);
#endif
#if BLUE
        postProcessColor = glm::vec3(0.5f, 0.5f, 1.f);
#endif
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) {
            pathSegments[idx].color *= postProcessColor;
        }
    }
}
/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */

int thrustCompaction(int num, PathSegment* dev_path) {
    //int new_num_paths = 0;
    //auto input = thrust::device_ptr<PathSegment>(dev_path);
    ////thrust::device_vector<PathSegment> input(dev_path, dev_path + num);
    //auto newEnd = thrust::remove_if(input, input+num, is_not_zero());
    //for (auto a = input; a != newEnd; a++) {
    //        new_num_paths++;
    //}
    int new_num_paths;
    //PathSegment* newEnd = thrust::remove_if(dev_path, dev_path + num, is_not_zero());
    PathSegment* newEnd = thrust::partition(thrust::device, dev_path, dev_path + num, is_not_zero());
    new_num_paths = newEnd - dev_path;
    return new_num_paths;
}

void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing
    generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");
    
    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
    int new_num_paths = num_paths;
    bool iterationComplete = false;
    bool firstRoundComplete = false;
    while (!iterationComplete) {
        // clean shading chunks if first iteration
        if (firstRoundComplete && CACHE_INTERSECTION) {
            cudaMemcpy(dev_intersections, dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyHostToHost);
        }
        else {
            cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
        }

        // tracing
        dim3 numblocksPathSegmentTracing = (new_num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , new_num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_intersections
            , dev_primitives
            , dev_materials
            , dev_texData
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

#if SORT_RAY
        thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections(dev_intersections);
        thrust::device_ptr<PathSegment> dev_thrust_segment = thrust::device_ptr<PathSegment>(dev_paths);
        thrust::sort_by_key(thrust::device, dev_thrust_intersections, dev_thrust_intersections + new_num_paths, dev_thrust_segment, compareMatId());
#endif
        //
        // 
        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
          // evaluating the BSDF.
          // Start off with just a big kernel that handles all the different
          // materials you have in the scenefile.
          // TODO: compare between directly shading the path segments and shading
          // path segments that have been reshuffled to be contiguous in memory.
        shadeDirectLight << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            new_num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures,
            dev_texData,
            dev_lightPoint
            );
        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            new_num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures,
            dev_texData,
            dev_lightPoint,
            dev_cudaTexObjs
            );
#if POSTPROCESS
        shadePostProcess << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            new_num_paths,
            dev_intersections,
            dev_paths
            );
#endif
        if (!firstRoundComplete && CACHE_INTERSECTION) {
            firstRoundComplete = true;
            cudaMemcpy(dev_cache_intersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyHostToHost);
            cudaDeviceSynchronize();
        }//cache process

        //Stream Compaction
        new_num_paths = thrustCompaction(new_num_paths, dev_paths);
        if (new_num_paths == 0) {
            iterationComplete = true;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}


