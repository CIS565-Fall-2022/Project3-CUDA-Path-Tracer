#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>

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

#define BVH_DEBUG_VISUALIZATION false

int ToneMapping::method = ToneMapping::ACES;

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* Image, int toneMapping) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);

		// Tonemapping and gamma correction
		glm::vec3 color = Image[index] / float(iter);

		switch (toneMapping) {
		case ToneMapping::Filmic:
			color = Math::filmic(color);
			break;
		case ToneMapping::ACES:
			color = Math::ACES(color);
			break;
		case ToneMapping::None:
			break;
		}
		color = Math::correctGamma(color);
		glm::ivec3 iColor = glm::clamp(glm::ivec3(color * 255.f), glm::ivec3(0), glm::ivec3(255));

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = iColor.x;
		pbo[index].y = iColor.y;
		pbo[index].z = iColor.z;
	}
}

#define PixelIdxForTerminated -1

static Scene* hstScene = nullptr;
static GuiDataContainer* guiData = nullptr;
static glm::vec3* devImage = nullptr;
static PathSegment* devPaths = nullptr;
static PathSegment* devTerminatedPaths = nullptr;
static Intersection* devIntersections = nullptr;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static thrust::device_ptr<PathSegment> devPathsThr;
static thrust::device_ptr<PathSegment> devTerminatedPathsThr;
 
void InitDataContainer(GuiDataContainer* imGuiData) {
	guiData = imGuiData;
}

void pathTraceInit(Scene* scene) {
	hstScene = scene;

	const Camera& cam = hstScene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&devImage, pixelcount * sizeof(glm::vec3));
	cudaMemset(devImage, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&devPaths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&devTerminatedPaths, pixelcount * sizeof(PathSegment));
	devPathsThr = thrust::device_ptr<PathSegment>(devPaths);
	devTerminatedPathsThr = thrust::device_ptr<PathSegment>(devTerminatedPaths);

	cudaMalloc(&devIntersections, pixelcount * sizeof(Intersection));
	cudaMemset(devIntersections, 0, pixelcount * sizeof(Intersection));

	checkCUDAError("pathTraceInit");
}

void pathTraceFree() {
	cudaFree(devImage);  // no-op if devImage is null
	cudaFree(devPaths);
	cudaFree(devTerminatedPaths);
	cudaFree(devIntersections);
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
		glm::vec3 pFocusPlane = glm::vec3(ruv * glm::vec2(aspect, 1.f) * tanFovY, 1.f) * cam.focalDist;
		glm::vec3 dir = pFocusPlane - pLens;
		dir = glm::normalize(glm::mat3(cam.right, cam.up, cam.view) * dir);

		segment.ray.origin = cam.position + cam.right * pLens.x + cam.up * pLens.y;
		segment.ray.direction = dir;

		segment.throughput = glm::vec3(1.f);
		segment.radiance = glm::vec3(0.f);
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth,
	int numPaths,
	PathSegment* pathSegments,
	DevScene* scene,
	Intersection* intersections
) {
	int pathIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pathIdx < numPaths) {
#if BVH_DEBUG_VISUALIZATION
	scene->visualizedIntersect(pathSegments[pathIdx].ray, intersections[pathIdx]);
#else
	Intersection intersec;
	PathSegment segment = pathSegments[pathIdx];
	scene->intersect(segment.ray, intersec);

	if (intersec.primId != NullPrimitive) {
		if (scene->devMaterials[intersec.matId].type == Material::Type::Light) {
#if SCENE_LIGHT_SINGLE_SIDED
			if (glm::dot(intersec.norm, segment.ray.direction) < 0.f) {
				intersec.primId = NullPrimitive;
			}
			else
#endif
			if (depth != 0) {
				// If not first ray, preserve previous sampling information for
				// MIS calculation
				intersec.prevPos = segment.ray.origin;
				// intersec.prevBSDFPdf = segment.BSDFPdf;
			}
		}
		else {
			intersec.wo = -segment.ray.direction;
		}
	}
	intersections[pathIdx] = intersec;
#endif
	}
}

__global__ void computeTerminatedRays(
	int depth,
	PathSegment* segments,
	Intersection* intersections,
	DevScene* scene,
	int numPaths
) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= numPaths) {
		return;
	}
}

__global__ void pathIntegSampleSurface(
	int iter,
	int depth,
	PathSegment* segments,
	Intersection* intersections,
	DevScene* scene,
	int numPaths
) {
	const int SamplesConsumedOneIter = 10;

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= numPaths) {
		return;
	}
	Intersection intersec = intersections[idx];

	if (intersec.primId == NullPrimitive) {
		// TODO
		// Environment map
		if (Math::luminance(segments[idx].radiance) < 1e-4f) {
			segments[idx].pixelIndex = PixelIdxForTerminated;
		}
		else {
			segments[idx].remainingBounces = 0;
		}
		return;
	}

#if BVH_DEBUG_VISUALIZATION
	float logDepth = 0.f;
	int size = scene->BVHSize;
	while (size) {
		logDepth += 1.f;
		size >>= 1;
	}
	segment.radiance = glm::vec3(float(intersec.primId) / logDepth * .1f);
	//segment.radiance = intersec.primitive > 16 ? glm::vec3(1.f) : glm::vec3(0.f);
	segment.remainingBounces = 0;
	return;
#endif

	PathSegment& segment = segments[idx];
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 4 + depth * SamplesConsumedOneIter);

	Material material = scene->devMaterials[intersec.matId];
	glm::vec3 accRadiance(0.f);

	if (material.type == Material::Type::Light) {
		glm::vec3 radiance = material.baseColor * material.emittance;
		if (depth == 0) {
			accRadiance += radiance;
		}
		else if (segment.deltaSample) {
			accRadiance += radiance * segment.throughput;
		}
		else {
			float lightPdf = Math::pdfAreaToSolidAngle(Math::luminance(radiance) * scene->sumLightPowerInv,
				intersec.prevPos, intersec.pos, intersec.norm);
			float BSDFPdf = segment.BSDFPdf;
			accRadiance += radiance * segment.throughput * Math::powerHeuristic(BSDFPdf, lightPdf);
		}
		segment.remainingBounces = 0;
	}
	else {
		bool deltaBSDF = (material.type == Material::Type::Dielectric);
		if (material.type != Material::Type::Dielectric && glm::dot(intersec.norm, intersec.wo) < 0.f) {
			intersec.norm = -intersec.norm;
		}

		if (!deltaBSDF) {
			glm::vec3 radiance;
			glm::vec3 wi;
			float lightPdf = scene->sampleDirectLight(intersec.pos, sample4D(rng), radiance, wi);

			if (lightPdf > 0.f) {
				float BSDFPdf = material.pdf(intersec.norm, intersec.wo, wi);
				accRadiance += segment.throughput * material.BSDF(intersec.norm, intersec.wo, wi) *
					radiance * Math::satDot(intersec.norm, wi) / lightPdf * Math::powerHeuristic(lightPdf, BSDFPdf);
			}
		}

		BSDFSample sample;
		material.sample(intersec.norm, intersec.wo, sample3D(rng), sample);

		if (sample.type == BSDFSampleType::Invalid) {
			// Terminate path if sampling fails
			segment.remainingBounces = 0;
		}
		else {
			bool deltaSample = (sample.type & BSDFSampleType::Specular);
			segment.throughput *= sample.bsdf / sample.pdf *
				(deltaSample ? 1.f : Math::absDot(intersec.norm, sample.dir));
			segment.ray = makeOffsetedRay(intersec.pos, sample.dir);
			segment.BSDFPdf = sample.pdf;
			segment.deltaSample = deltaSample;
			segment.remainingBounces--;
		}
	}
	//if (depth == 1)
	segment.radiance += accRadiance;
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths) {
		PathSegment iterationPath = iterationPaths[index];
		if (iterationPath.pixelIndex >= 0 && iterationPath.remainingBounces <= 0) {
			image[iterationPath.pixelIndex] += iterationPath.radiance;
		}
	}
}

struct CompactTerminatedPaths {
	__host__ __device__ bool operator() (const PathSegment& segment) {
		return !(segment.pixelIndex >= 0 && segment.remainingBounces <= 0);
	}
};

struct RemoveInvalidPaths {
	__host__ __device__ bool operator() (const PathSegment& segment) {
		return segment.pixelIndex < 0 || segment.remainingBounces <= 0;
	}
};

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
	checkCUDAError("PT::generateRayFromCamera");
	cudaDeviceSynchronize();

	int depth = 0;
	int numPaths = pixelCount;

	auto devTerminatedThr = devTerminatedPathsThr;

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
			hstScene->devScene,
			devIntersections
		);
		checkCUDAError("PT::computeInteractions");
		cudaDeviceSynchronize();

		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		pathIntegSampleSurface<<<numBlocksPathSegmentTracing, blockSize1D>>>(
			iter, depth, devPaths, devIntersections, hstScene->devScene, numPaths
		);
		checkCUDAError("PT::sampleSurface");
		cudaDeviceSynchronize();

		// Compact paths that are terminated but carry contribution into a separate buffer
		devTerminatedThr = thrust::remove_copy_if(devPathsThr, devPathsThr + numPaths, devTerminatedThr, CompactTerminatedPaths());
		// Only keep active paths
		auto end = thrust::remove_if(devPathsThr, devPathsThr + numPaths, RemoveInvalidPaths());
		numPaths = end - devPathsThr;
		//std::cout << "Remaining paths: " << numPaths << "\n";

		iterationComplete = (numPaths == 0);
		depth++;

		if (guiData != nullptr) {
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelCount + blockSize1D - 1) / blockSize1D;
	int numContributing = devTerminatedThr.get() - devTerminatedPaths;
	finalGather<<<numBlocksPixels, blockSize1D>>>(numContributing, devImage, devTerminatedPaths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid2D, blockSize2D>>>(pbo, cam.resolution, iter, devImage, ToneMapping::method);

	// Retrieve image from GPU
	cudaMemcpy(hstScene->state.image.data(), devImage,
		pixelCount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathTrace");
}