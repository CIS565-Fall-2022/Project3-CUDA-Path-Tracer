#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include <device_launch_parameters.h>

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

///////////////////////////////////////
// --- toggleable things ---
#define _STREAM_COMPACTION_			0
#define _GROUP_RAYS_BY_MATERIAL_	0
#define _CACHE_FIRST_BOUNCE_		0
// --- end toggleable things ---

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
#if _CACHE_FIRST_BOUNCE_
static ShadeableIntersection* dev_first_isecs = NULL;
#endif

// predicate for thrust::remove_if stream compaction
struct partition_terminated_paths {
	__host__ __device__
	bool operator()(const PathSegment& p) { return p.remainingBounces > 0; }
};

// _GROUP_RAYS_BY_MATERIAL_ sorting comparison
struct compare_intersection_mat {
	__host__ __device__
	bool operator()(const ShadeableIntersection& i1, const ShadeableIntersection& i2) {
		return i1.materialId < i2.materialId;
	}
};

void InitDataContainer(GuiDataContainer* imGuiData) { guiData = imGuiData; }

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

#if _CACHE_FIRST_BOUNCE_
	cudaMalloc(&dev_first_isecs, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_first_isecs, 0, pixelcount * sizeof(ShadeableIntersection));
#endif
	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);

#if _CACHE_FIRST_BOUNCE_
	cudaFree(dev_first_isecs);
#endif
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
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
__global__ void computeIntersections(
	int depth,
	int num_paths,
	PathSegment* pathSegments,
	Geom* geoms,
	int geoms_size,
	ShadeableIntersection* intersections) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index < num_paths) {
		PathSegment pathSegment = pathSegments[path_index];

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

			if (geom.type == CUBE) {
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE) {
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t) {
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1) {
			intersections[path_index].t = -1.0f;
		}
		else {
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

// Shade path segments based on intersections, generate new rays w/ BSDF.
__global__ void shadeMaterial (
	int iter,
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials,
	int depth) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
#if _STREAM_COMPACTION_
#else
	if (pathSegments[idx].remainingBounces == 0) { return; }
#endif
	if (idx < num_paths) {
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			Material material = materials[intersection.materialId];
			if (material.emittance > 0.0f) { // material is bright; "light" the ray
				pathSegments[idx].color *= (material.color * material.emittance);
				pathSegments[idx].remainingBounces = 0; // terminate if hit light
			} else {
				// BSDF evaluation: in-place assigns new direction + color for ray
				scatterRay(pathSegments[idx],
					pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction, // where ray intersects material
					intersection.surfaceNormal,
					material,
					makeSeededRandomEngine(iter, idx, depth));
				if (--pathSegments[idx].remainingBounces == 0) {
					// seems to look better to set the non-terminataing to black
					pathSegments[idx].color = glm::vec3(0.0f); 
				}
			}
		}
		else {
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0; // terminate if hit nothing
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < nPaths) {
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
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
	// --- Start generate array of path rays (that come out of the camera)
	//   * Each path ray has (ray, color) pair,
	//   where color starts as the multiplicative identity, white = (1, 1, 1).
	generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
	// --- End generate rays
	
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;
	bool iterationComplete = false;
	while (!iterationComplete) {
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		
		// --- start compute intersection ---
	//   * Compute an intersection in the scene for each path ray.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if _CACHE_FIRST_BOUNCE_
		if (iter > 1 && depth == 0) {
			cudaMemcpy(dev_intersections,
				dev_first_isecs,
				pixelcount * sizeof(ShadeableIntersection),
				cudaMemcpyDeviceToDevice);
		} else {
#endif
		computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
			depth,
			num_paths,
			dev_paths,
			dev_geoms,
			hst_scene->geoms.size(),
			dev_intersections);
		checkCUDAError("trace one bounce");
#if _CACHE_FIRST_BOUNCE_
		}
		if (iter == 1 && depth == 0) {
			cudaMemcpy(dev_first_isecs,
				dev_intersections,
				pixelcount * sizeof(ShadeableIntersection),
				cudaMemcpyDeviceToDevice);
		}
#endif
		cudaDeviceSynchronize();
		depth++;
		// --- end compute intersection --- 

#if _GROUP_RAYS_BY_MATERIAL_
		thrust::sort_by_key(thrust::device,
			dev_intersections, dev_intersections + num_paths,
			dev_paths,
			compare_intersection_mat());
#endif
		// can't use a 2D kernel launch any more - switch to 1D.
		// --- begin Shading Stage ---
		shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			depth);
		// --- end shading stage
		
		// --- begin stream compaction for all of the terminated paths
		//	   determine if iterations should end (ie all paths done)
#if _STREAM_COMPACTION_
		dev_path_end = thrust::partition(thrust::device,
			dev_paths, dev_path_end,
			partition_terminated_paths()); // overloaded struct; lambdas need special compilation flag
		num_paths = dev_path_end - dev_paths;
		iterationComplete = (num_paths == 0);
#else
		iterationComplete = (depth == traceDepth);
#endif
		// --- end stream compaction

		if (guiData != NULL) {
			guiData->TracedDepth = depth;
		}
	}
	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	checkCUDAError("pathtrace");
}