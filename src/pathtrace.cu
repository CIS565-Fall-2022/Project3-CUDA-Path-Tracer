#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "material.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "mathUtil.h"
#include "sampler.h"

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

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);

		// ACES tonemapping and gamma correction
		glm::vec3 color = image[index] / float(iter);
		glm::vec3 mapped = Math::correctGamma(Math::ACES(color));
		glm::ivec3 iColor = glm::clamp(glm::ivec3(mapped * 255.f), glm::ivec3(0), glm::ivec3(255));

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = iColor.x;
		pbo[index].y = iColor.y;
		pbo[index].z = iColor.z;
	}
}

static Scene* hstScene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* devImage = NULL;
static Geom* devGeoms = NULL;
static Material* devMaterials = NULL;
static PathSegment* devPaths = NULL;
static Intersection* devIntersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
 
void InitDataContainer(GuiDataContainer* imGuiData) {
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
	hstScene = scene;

	const Camera& cam = hstScene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&devImage, pixelcount * sizeof(glm::vec3));
	cudaMemset(devImage, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&devPaths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&devGeoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(devGeoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&devMaterials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(devMaterials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&devIntersections, pixelcount * sizeof(Intersection));
	cudaMemset(devIntersections, 0, pixelcount * sizeof(Intersection));

	// TODO: initialize any extra device memeory you need

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {

	cudaFree(devImage);  // no-op if devImage is null
	cudaFree(devPaths);
	cudaFree(devGeoms);
	cudaFree(devMaterials);
	cudaFree(devIntersections);
	// TODO: clean up any extra device memory you created

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments) {

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		glm::vec4 r = sample4D(rng);

		PathSegment& segment = pathSegments[index];

		// Antialiasing and physically based camera (lens effect)

		float aspect = float(cam.resolution.x) / cam.resolution.y;
		float tanFovY = glm::tan(glm::radians(cam.fov.y));
		glm::vec2 pixelSize = 1.f / glm::vec2(cam.resolution);
		glm::vec2 scr = glm::vec2(x, y) * pixelSize;
		glm::vec2 ruv = scr + pixelSize * glm::vec2(r.x, r.y);
		ruv = 1.f - ruv * 2.f;

		glm::vec3 pLens = glm::vec3(Math::toConcentricDisk(r.z, r.w) * cam.lensRadius, 0.f);
		glm::vec3 pFocusPlane = glm::vec3(ruv * glm::vec2(aspect, 1.f) * cam.focalDist * tanFovY, cam.focalDist);
		glm::vec3 dir = pFocusPlane - pLens;
		dir = glm::normalize(glm::mat3(cam.right, cam.up, cam.view) * dir);

		segment.ray.origin = cam.position + cam.right * pLens.x + cam.up * pLens.y;
		segment.ray.direction = dir;

		segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
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
	Geom* geoms,
	int geoms_size,
	Intersection* intersections
) {
	int pathIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pathIdx < num_paths) {
		PathSegment pathSegment = pathSegments[pathIdx];

		float dist;
		glm::vec3 intersectPoint;
		glm::vec3 normal;
		float tMin = FLT_MAX;
		int hitGeomIdx = -1;
		bool outside = true;

		glm::vec3 tmpIntersect;
		glm::vec3 tmpNormal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++) {
			Geom& geom = geoms[i];
			// TODO: add more intersection tests here... triangle? metaball? CSG?
			dist = intersectGeom(geom, pathSegment.ray, tmpIntersect, tmpNormal, outside);
			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (dist > 0.0f && tMin > dist) {
				tMin = dist;
				hitGeomIdx = i;
				intersectPoint = tmpIntersect;
				normal = tmpNormal;
			}
		}

		if (hitGeomIdx == -1) {
			intersections[pathIdx].dist = -1.0f;
		}
		else {
			//The ray hits something
			intersections[pathIdx].dist = tMin;
			intersections[pathIdx].materialId = geoms[hitGeomIdx].materialId;
			intersections[pathIdx].surfaceNormal = normal;
			intersections[pathIdx].position = intersectPoint;
			intersections[pathIdx].incomingDir = -pathSegment.ray.direction;
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
	int iter, 
	int num_paths, 
	Intersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		Intersection intersection = shadeableIntersections[idx];
		if (intersection.dist > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.baseColor;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].throughput *= (materialColor * material.emittance);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				pathSegments[idx].throughput *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.dist * 0.02f) * materialColor) * 0.7f;
				pathSegments[idx].throughput *= u01(rng); // apply some noise because why not
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].throughput = glm::vec3(0.0f);
		}
	}
}

__global__ void pathIntegSampleSurface(
	int iter,
	PathSegment* segments,
	Intersection* intersections,
	Material* materials,
	int numPaths
) {
	const int SamplesConsumedOneIter = 1;

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= numPaths) {
		return;
	}
	Intersection intersec = intersections[idx];
	if (intersec.dist < 0) {
		return;
	}

	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 4 + iter * SamplesConsumedOneIter);
	Material material = materials[intersec.materialId];

	// TODO
	// Perform light area sampling and MIS

	if (material.type == Material::Type::Light) {
		// TODO
		// MIS

		segments[idx].throughput *= material.baseColor * material.emittance;
		segments[idx].remainingBounces = 0;
	}
	else {
		BSDFSample sample;
		materialSample(intersec.surfaceNormal, intersec.incomingDir, material, sample3D(rng), sample);

		if (sample.pdf < 0) {
			// Terminate path if sampling fails
			segments[idx].remainingBounces = 0;
			segments[idx].pixelIndex = -1;
		}
		else {
			bool isSampleDelta = sample.type & BSDFSampleType::Specular;
			segments[idx].throughput *= sample.bsdf / sample.pdf *
				(isSampleDelta ? 1.f : Math::absDot(intersec.surfaceNormal, sample.dir));
			segments[idx].ray = makeRay(intersec.position, sample.dir);
			segments[idx].remainingBounces--;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths) {
		PathSegment iterationPath = iterationPaths[index];
		if (iterationPaths[index].remainingBounces == 0) {
			image[iterationPath.pixelIndex] += iterationPath.throughput;
		}
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathTrace(uchar4* pbo, int frame, int iter) {
	const int traceDepth = hstScene->state.traceDepth;
	const Camera& cam = hstScene->state.camera;
	const int pixelCount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2D(8, 8);
	const dim3 blocksPerGrid2D(
		(cam.resolution.x + blockSize2D.x - 1) / blockSize2D.x,
		(cam.resolution.y + blockSize2D.y - 1) / blockSize2D.y);

	// 1D block for path tracing
	const int blockSize1D = 128;

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

	generateRayFromCamera<<<blocksPerGrid2D, blockSize2D>>>(cam, iter, traceDepth, devPaths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* devPathEnd = devPaths + pixelCount;
	int numPaths = devPathEnd - devPaths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {
		// clean shading chunks
		cudaMemset(devIntersections, 0, pixelCount * sizeof(Intersection));

		// tracing
		dim3 numBlocksPathSegmentTracing = (numPaths + blockSize1D - 1) / blockSize1D;
		computeIntersections<<<numBlocksPathSegmentTracing, blockSize1D>>>(
			depth, 
			numPaths,
			devPaths, 
			devGeoms,
			hstScene->geoms.size(), 
			devIntersections
		);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		//shadeFakeMaterial<<<numBlocksPathSegmentTracing, blockSize1D>>>(
		//	iter,
		//	numPaths,
		//	devIntersections,
		//	devPaths,
		//	devMaterials
		//);
		//iterationComplete = true; // TODO: should be based off stream compaction results.

		pathIntegSampleSurface<<<numBlocksPathSegmentTracing, blockSize1D>>>(
			iter, devPaths, devIntersections, devMaterials, numPaths
		);

		thrust::remove_if(devPaths.

		iterationComplete = true;

		if (guiData != NULL) {
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelCount + blockSize1D - 1) / blockSize1D;
	finalGather<<<numBlocksPixels, blockSize1D>>>(numPaths, devImage, devPaths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid2D, blockSize2D>>>(pbo, cam.resolution, iter, devImage);

	// Retrieve image from GPU
	cudaMemcpy(hstScene->state.image.data(), devImage,
		pixelCount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathTrace");
}