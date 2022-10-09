#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "setting.h"
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"


#if not STD_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include <stb_image.h>


#define ERRORCHECK 1



#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
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


struct PathPartition
{
    __host__ __device__ bool operator()(const PathSegment& p)
    {
        return p.remainingBounces > 0;
        
        //return !(p.remainingBounces <= 0 || p.hitLightSource ||
        //    (p.color[0] < EPSILON && p.color[1] < EPSILON && p.color[2] < EPSILON));
    }
}pathPartition;

struct MaterialSort
{
    __host__ __device__ bool operator()(const ShadeableIntersection& s1, const ShadeableIntersection& s2)
    {
        return s1.materialId < s2.materialId;
    }
}materialSort;



__host__ __device__ glm::vec2 DirectionToSpereUV(glm::vec3 dir)
{
    float phi = glm::atan(dir.z, dir.x);
    if (phi < 0)
    {
        phi += TWO_PI;
    }

    float theta = glm::acos(dir.y);
    return glm::vec2(1 - phi / TWO_PI, 1 - theta / PI);
}

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
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc

// Mesh
static Triangle* dev_triangles = NULL;

// Texture map
static TextureInfo* dev_textures = NULL;
static glm::vec3* dev_pixels = NULL;

// Normal map
static TextureInfo* dev_textureNormals = NULL;
static glm::vec3* dev_normals = NULL;

// BVH
static int* dev_bvhIndexArrayToUse = NULL;
static LinearBVHNode* dev_bvhNodes = NULL;

// Skybox texture
static TextureInfo* dev_skyBbxTexture = NULL;
static glm::vec3* dev_skyboxPixels = NULL;

// Cache first bounce
#if CACHE_FIRST_INTERSECTIONS
static ShadeableIntersection* dev_intersectionsCache = NULL;
#endif




void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

// ============== Texture related =====================

void LoadTexturesToDevice(Scene* scene)
{
    std::cout << "Load Textures to device starts..." << std::endl;

#if ENABLE_TEXTURE
    // Load texture into CPU
    int totalPixelCount = 0;
    std::vector<TextureInfo> textures;
    std::vector<glm::vec3> pixels;
    for (int i = 0; i < scene->textureIds.size(); ++i)
    {
        int width, height, channels;
        stbi_hdr_to_ldr_gamma(1.0f);
        unsigned char* img = stbi_load(scene->textureIds[i].c_str(), &width, &height, &channels, 3);

        if (img == NULL)
        {
            cout << "Load texture [" << scene->textureIds[i] << "] fails." << endl;
            continue;
        }

        TextureInfo texture;
        texture.id = scene->textureIds[i].c_str();
        texture.width = width;
        texture.height = height;
        texture.channels = channels;
        texture.startIndex = totalPixelCount;
        textures.push_back(texture);

        totalPixelCount += width * height;

        // Read each pixel
        for (int j = 0; j < width * height; ++j)
        {
            pixels.emplace_back(glm::vec3(img[3 * j + 0] / 255.0f, img[3 * j + 1] / 255.0f, img[3 * j + 2] / 255.0f));
        }
        stbi_image_free(img);
    }

    // Send Texture to GPU
    cudaMalloc((void**)&dev_textures, textures.size() * sizeof(TextureInfo));
    cudaMemcpy(dev_textures, textures.data(), textures.size() * sizeof(TextureInfo), cudaMemcpyHostToDevice);
    checkCUDAError("Copy texture infos to device");

    cudaMalloc((void**)&dev_pixels, totalPixelCount * sizeof(glm::vec3));
    cudaMemcpy(dev_pixels, pixels.data(), totalPixelCount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    checkCUDAError("Copy pixels to device");

#endif

#if ENABLE_NORMAL_MAP
    // Load normal map into CPU
    int totalNormalCount = 0;
    std::vector<TextureInfo> textureNormals;
    std::vector<glm::vec3> normals;
    for (int i = 0; i < scene->textureNormalIds.size(); ++i)
    {
        int width, height, channels;
        stbi_hdr_to_ldr_gamma(1.0f);
        unsigned char* img = stbi_load(scene->textureNormalIds[i].c_str(), &width, &height, &channels, 3);

        if (img == NULL)
        {
            cout << "Load texture normal [" << scene->textureNormalIds[i] << "] fails." << endl;
            continue;
        }

        TextureInfo normalMap;
        normalMap.id = scene->textureNormalIds[i].c_str();
        normalMap.width = width;
        normalMap.height = height;
        normalMap.channels = channels;
        normalMap.startIndex = totalNormalCount;
        textureNormals.push_back(normalMap);

        totalNormalCount += width * height;

        // Read normals
        for (int j = 0; j < width * height; ++j)
        {
            glm::vec3 rgb(img[3 * j + 0] / 255.0f, img[3 * j + 1] / 255.0f, img[3 * j + 2] / 255.0f);
            normals.emplace_back(rgb * 2.0f - glm::vec3(1.0f, 1.0f, 1.0f));
        }
        stbi_image_free(img);  
    }

    // Send Texture normal to GPU
    cudaMalloc((void**)&dev_textureNormals, textureNormals.size() * sizeof(TextureInfo));
    cudaMemcpy(dev_textureNormals, textureNormals.data(), textureNormals.size() * sizeof(TextureInfo), cudaMemcpyHostToDevice);
    checkCUDAError("Copy texture normal infos to device");

    cudaMalloc((void**)&dev_normals, normals.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_normals, normals.data(), normals.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    checkCUDAError("Copy normals to device");

#endif

    cudaDeviceSynchronize();

    std::cout << "Load texture to device done" << std::endl;
}

void FreeTextures()
{
    std::cout << "Free Texures " << std::endl;
    
#if ENABLE_TEXTURE
    cudaFree(dev_textures);
    cudaFree(dev_pixels);
#endif

#if ENABLE_NORMAL_MAP
    cudaFree(dev_textureNormals);
    cudaFree(dev_normals);
#endif
}

__host__ __device__ int getTextureElementIndex(const TextureInfo&tex, glm::vec2 uv)
{
    // UV to pixel space
    int w = tex.width * uv[0] - 0.5;
    int h = tex.height * (1 - uv[1]) - 0.5;

    // 2D pixel space to 1D
    int pixelIndex = h * tex.width + w;

    return pixelIndex + tex.startIndex;
}


// ====================================================

// ============== BVH related =========================

void LoadBVHToDevice(Scene* scene)
{
    std::cout << "Loading BVH nodes to device." << std::endl;

    cudaMalloc((void**)&dev_bvhNodes, scene->bvhNodes.size() * sizeof(LinearBVHNode));
    cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(LinearBVHNode), cudaMemcpyHostToDevice);

    int pixelCount = scene->state.camera.resolution.x * scene->state.camera.resolution.y;
    cudaMalloc((void**)&dev_bvhIndexArrayToUse, BVH_INTERSECT_STACK_SIZE * pixelCount * sizeof(int));
    cudaMemset(dev_bvhIndexArrayToUse, -1, BVH_INTERSECT_STACK_SIZE * pixelCount * sizeof(int));
}

void FreeBVH()
{
    std::cout << "Free BVH" << std::endl;

    cudaFree(dev_bvhNodes);
    cudaFree(dev_bvhIndexArrayToUse);
}

// ====================================================

void LoadSkyboxTextureToDevice(Scene* scene)
{
    int width, height, channels;
    stbi_hdr_to_ldr_gamma(1.0f);
    unsigned char* img = stbi_load(scene->skyboxId.c_str(), &width, &height, &channels, 3);

    if (img == NULL)
    {
        cout << "Load skybox texture normal [" << scene->skyboxId << "] fails." << endl;
        return;
    }

    TextureInfo skyboxTextureInfo;
    skyboxTextureInfo.id = scene->skyboxId.c_str();
    skyboxTextureInfo.width = width;
    skyboxTextureInfo.height = height;
    skyboxTextureInfo.channels = channels;
    skyboxTextureInfo.startIndex = 0;

    int totalPixelCount = width * height;

    std::vector<glm::vec3> pixels;
    for (int i = 0; i < totalPixelCount; ++i)
    {
        pixels.emplace_back(glm::vec3(img[3 * i + 0] / 255.0f, img[3 * i + 1] / 255.0f, img[3 * i + 2] / 255.0f));
    }
    stbi_image_free(img);

    cudaMalloc((void**)&dev_skyBbxTexture, sizeof(TextureInfo));
    cudaMemcpy(dev_skyBbxTexture, &skyboxTextureInfo, sizeof(TextureInfo), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_skyboxPixels, totalPixelCount * sizeof(glm::vec3));
    cudaMemcpy(dev_skyboxPixels, pixels.data(), totalPixelCount * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    checkCUDAError("Load Skybox To Device");

    std::cout << "Skybox texture [" << scene->skyboxId << "] loaded." << std::endl;
}

void FreeSkyboxTexure()
{
    cudaFree(dev_skyBbxTexture);
    cudaFree(dev_skyboxPixels);
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
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

	// TODO: initialize any extra device memeory you need

#if CACHE_FIRST_INTERSECTIONS
    cudaMalloc((void**)&dev_intersectionsCache, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersectionsCache, 0, pixelcount * sizeof(ShadeableIntersection));
#endif

    cudaMalloc((void**)&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    //std::cout << "copy [" << scene->triangles.size() << "] triangles to device" << std::endl;

    //for (int i = 0; i < scene->geoms.size(); ++i)
    //{
    //    std::cout << "geom bvh start index = " << scene->geoms[i].bvhNodeStartIndex << std::endl;
    //}

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {

    cudaDeviceSynchronize();

	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);

	// TODO: clean up any extra device memory you created

#if CACHE_FIRST_INTERSECTIONS
    cudaFree(dev_intersectionsCache);
#endif

    cudaFree(dev_triangles);

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

    if (x < cam.resolution.x && y < cam.resolution.y)
    {
        int index = x + (y * cam.resolution.x);
        PathSegment & segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
#if ENABLE_ANTI_ALIASING

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, pathSegments->remainingBounces);
        thrust::uniform_real_distribution<float> r(-0.5f, 0.5f);
        float xRandom = r(rng);
        float yRandom = r(rng);

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + xRandom - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + yRandom - (float)cam.resolution.y * 0.5f)
        );
#else
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
#endif

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
    , PathSegment* pathSegments
    , Geom* geoms
    , Triangle* triangles
    , LinearBVHNode* bvhNodes
    , int* bvhIndexArrayToUse
	, int geoms_size
	, ShadeableIntersection* intersections
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
        glm::vec2 uv(0.f);
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
        int triangleId = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
	
        // naive parse through global geoms
        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == OBJECT)
            {
                t = objIntersectionTest(geom, pathSegment.ray, triangles, bvhNodes, bvhIndexArrayToUse, path_index,
                    tmp_intersect, tmp_normal, uv, triangleId, outside);
            }
            else
            {
                continue;
            }
            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
            intersections[path_index].uv = uv;
            intersections[path_index].geomId = hit_geom_index;
            intersections[path_index].triangleId = triangleId;
		}

	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter
    , int depth
    , int traceDepth
    , int num_paths
    , Geom* geoms
    , Triangle* triangles
    , ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments
    , Material* materials
    , TextureInfo* textures
    , glm::vec3* pixels
    , TextureInfo* textureNormals
    , glm::vec3* normals
    , TextureInfo* skyboxInfo
    , glm::vec3* skyboxPixels
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= materialColor * material.emittance;
                pathSegments[idx].hitLightSource = true;    // Terminate light
                pathSegments[idx].remainingBounces = -1;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {

                glm::vec3 intersect = getPointOnRay(pathSegments[idx].ray, intersection.t);

                // Get texture color
                glm::vec3 textureColor(1.0f, 1.0f, 1.0f);
#if ENABLE_TEXTURE
                if (material.textureIndex >= 0)
                {
                    TextureInfo tex = textures[material.textureIndex];
                    int index = getTextureElementIndex(tex, intersection.uv);
                    textureColor = pixels[index];
                }
#endif

                // Get texture normal
                glm::vec3 normal = intersection.surfaceNormal;
#if ENABLE_NORMAL_MAP
                if (material.normalMapIndex >= 0)
                {
                    Geom geom = geoms[intersection.geomId];
                    TextureInfo norMap = textureNormals[material.normalMapIndex];
                    int index = getTextureElementIndex(norMap, intersection.uv);

                    //normal = ;    // Normal map normal in tangnet space

                    // Tangent space to model space
                    if (intersection.triangleId >= 0)
                    {
                        Triangle t = triangles[intersection.triangleId];

                        glm::vec3 deltaPos1 = t.v1 - t.v0;  // In model space
                        glm::vec3 deltaPos2 = t.v2 - t.v0;
                        glm::vec2 deltaUV1 = t.tex1 - t.tex0;
                        glm::vec2 deltaUV2 = t.tex2 - t.tex0;
                        
                        float r = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
                        glm::vec3 tagent = (deltaPos1 * deltaUV2.y - deltaPos2 * deltaUV1.y) * r;
                        glm::vec3 bitangent = (deltaPos2 * deltaUV1.x - deltaPos1 * deltaUV2.x) * r;
                        //glm::vec3 nor = glm::normalize(glm::cross(tagent, bitangent));
                        glm::vec3 nor = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(intersection.surfaceNormal, 0.0f)));
                        //glm::cross(tagent, bitangent);

                        //glm::mat3 TBN = glm::transpose(glm::mat3(tagent, bitangent, nor));
                        glm::mat3 TBN = glm::mat3(tagent, bitangent, nor);

                        normal = glm::normalize(TBN * normals[index]);
                    }

                    // Model space to world space
                    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(normal, 0.f)));
                }
#endif

                scatterRay(pathSegments[idx], intersect, normal,
                    material, textureColor, rng);
            }
        }
        else {
#if ENABLE_SKYBOX

            if (pathSegments[idx].isRefrectiveRay || depth == 0)
            {
                //printf("ok ");
                glm::vec2 uv = DirectionToSpereUV(pathSegments[idx].ray.direction);
                int index = getTextureElementIndex(*skyboxInfo, uv);
                pathSegments[idx].color *= skyboxPixels[index] * (pathSegments[idx].remainingBounces * 1.f) / (traceDepth * 1.f);
            }
#else
            pathSegments[idx].color *= glm::vec3(DEFAULT_SKY_COLOR);
#endif
            pathSegments[idx].remainingBounces = -1;
        }
    }
}



// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) 
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
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

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d>> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    thrust::device_ptr<PathSegment>thrust_paths(dev_paths);

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHE_FIRST_INTERSECTIONS
    
        if (depth == 0)
        {
            if (iter == 1)
            {
                // tracing            
                computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                    depth
                    , num_paths
                    , dev_paths
                    , dev_geoms
                    , dev_triangles
                    , dev_bvhNodes
                    , dev_bvhIndexArrayToUse
                    , hst_scene->geoms.size()
                    , dev_intersections
                    );
                checkCUDAError("trace one bounce");

                cudaMemcpy(dev_intersectionsCache, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            }
            else
            {
                cudaMemcpy(dev_intersections, dev_intersectionsCache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            }
        }
        else
        {
            // tracing
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , dev_triangles
                , dev_bvhNodes
                , dev_bvhIndexArrayToUse
                , hst_scene->geoms.size()
                , dev_intersections
                );
            checkCUDAError("trace one bounce");
        }
#else
        // tracing
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , dev_triangles
            , dev_bvhNodes
            , dev_bvhIndexArrayToUse
            , hst_scene->geoms.size()
            , dev_intersections
            );
        checkCUDAError("trace one bounce");

#endif
        cudaDeviceSynchronize();

        // Sort material by type so that ajecent threads will more likely to read the same memory
#if SORT_MATERIALS
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, materialSort);
#endif

        shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            depth,
            traceDepth,
            num_paths,
            dev_geoms,
            dev_triangles,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures,
            dev_pixels,
            dev_textureNormals,
            dev_normals,
            dev_skyBbxTexture,
            dev_skyboxPixels
            );
        checkCUDAError("trace one bounce");

        dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, pathPartition);
        num_paths = dev_path_end - dev_paths;
        
        depth++;
        iterationComplete = num_paths <= 0 || depth >= traceDepth;
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");

    //std::cout << "path trace one iteration" << std::endl;
}