#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <thrust/random.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

// impl switches
#define COMPACTION
// #define FAKE_SHADE

void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#ifndef NDEBUG
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

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(int iter, glm::vec3* pixs, uchar4* pbo, glm::ivec2 resolution) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = pixs[index];

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

static glm::vec3*             dev_image;
static Geom*                  dev_geoms;
static Material*              dev_materials;
static PathSegment*           dev_paths;
static ShadeableIntersection* dev_intersections;

// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	ALLOC(dev_image, pixelcount);
	ZERO(dev_image, pixelcount);

	ALLOC(dev_paths, pixelcount);

	ALLOC(dev_geoms, scene->geoms.size());
	H2D(dev_geoms, scene->geoms.data(), scene->geoms.size());
	
	ALLOC(dev_materials, scene->materials.size());
	H2D(dev_materials, scene->materials.data(), scene->materials.size());
	
	ALLOC(dev_intersections, pixelcount);
	ZERO(dev_intersections, pixelcount);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {

	FREE(dev_image);
	FREE(dev_paths);
	FREE(dev_geoms);
	FREE(dev_materials);
	FREE(dev_intersections);

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
		Ray cam_ray;
		cam_ray.origin = cam.position;
		
		// TODO: implement antialiasing by jittering the ray
		cam_ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);

		PathSegment& segment = pathSegments[index];
		segment.init(traceDepth, index, cam_ray);
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth,
	int num_paths,
	PathSegment* pathSegments,
	int geoms_size,
	Geom* geoms,
	ShadeableIntersection* intersections)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index >= num_paths) {
		return;
	}
	PathSegment pathSegment = pathSegments[path_index];

#ifndef COMPACTION
	if (!pathSegment.remainingBounces) {
		return;
	}
#endif // COMPACTION

	assert(pathSegment.remainingBounces > 0);

	float t;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	float t_min = FLT_MAX;
	int hit_geom_index = -1;
	bool outside = true;

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;

	// naive parse through global geoms

	for (int i = 0; i < geoms_size; i++) {
		Geom& geom = geoms[i];

		if (geom.type == CUBE)
		{
			t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
		} else if (geom.type == SPHERE)
		{
			t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
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
		}
	}

	if (hit_geom_index == -1)
	{
		intersections[path_index].t = -1.0f;
	} else
	{
		//The ray hits something
		intersections[path_index].t = t_min;
		intersections[path_index].materialId = geoms[hit_geom_index].materialid;
		intersections[path_index].surfaceNormal = normal;
	}
}

__global__ void shadeMaterial(
    int iter,
	int num_paths,
	PathSegment* paths,
	ShadeableIntersection* shadeableIntersections,
	Material* materials) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths) {
		return;
	}

	PathSegment& path = paths[idx];
	ShadeableIntersection intersection = shadeableIntersections[idx];

#ifndef COMPACTION
	if (!path.remainingBounces) {
		return;
	}
#endif // COMPACTION

	assert(path.remainingBounces > 0);

	if (intersection.t > 0.0f) {
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		Material material = materials[intersection.materialId];
		glm::vec3 materialColor = material.color;

		// If the material indicates that the object was a light, "light" the ray
		if (material.emittance > 0.0f) {
			path.color *= (materialColor * material.emittance);
			path.terminate();
		} else {
			glm::vec3 hit = intersection.t * path.ray.direction + path.ray.origin;
			scatterRay(path, hit, glm::normalize(intersection.surfaceNormal), material, rng);
		}
	} else {
		path.color = BACKGROUND_COLOR;
		path.terminate();
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
	, int num_paths
	, PathSegment* paths
	, ShadeableIntersection* shadeableIntersections
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		PathSegment& path = paths[idx];
		ShadeableIntersection intersection = shadeableIntersections[idx];

		assert(path.remainingBounces > 0);

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
				path.color *= (materialColor * material.emittance);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				path.color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				path.color *= u01(rng); // apply some noise because why not
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		} else {
			path.color = BACKGROUND_COLOR;
		}

		path.terminate();
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int numPixels, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < numPixels) {
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int const frame, int const iter) {
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

    generateRayFromCamera KERN_PARAM(blocksPerGrid2d, blockSize2d) (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
	for (int depth = 0, num_paths = pixelcount; num_paths > 0 && depth < traceDepth; ++depth) {
		
		// clean shading chunks
		MEMSET(dev_intersections, 0, num_paths);

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections KERN_PARAM(numblocksPathSegmentTracing, blockSize1d) (
			depth,
			num_paths,
			dev_paths,
			hst_scene->geoms.size(),
			dev_geoms,
			dev_intersections
		);

		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
#ifdef SORT_MAT

#endif

#ifdef FAKE_SHADE
#define shadeMaterial shadeFakeMaterial
#endif
		shadeMaterial KERN_PARAM(numblocksPathSegmentTracing, blockSize1d) (
			iter,
			num_paths,
			dev_paths,
			dev_intersections,
			dev_materials
		);

		checkCUDAError("shadeMaterial");
		cudaDeviceSynchronize();

#ifdef COMPACTION
		auto ptr = thrust::device_ptr<PathSegment>(dev_paths);
		num_paths = thrust::partition(ptr, ptr + num_paths, PathSegment::Pred()) - ptr;
#endif // COMPACTION

#ifdef MAX_DEPTH_OVERRIDE
		if (depth == MAX_DEPTH_OVERRIDE)
			break;
#endif
	}
	
	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather KERN_PARAM(numBlocksPixels, blockSize1d) (pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO KERN_PARAM(blocksPerGrid2d, blockSize2d) (iter, dev_image, pbo, cam.resolution);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}