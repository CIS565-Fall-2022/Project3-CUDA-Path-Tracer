#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <algorithm>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/functional.h>


#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "thrust/device_vector.h"
#include "thrust/remove.h"
#include "thrust/execution_policy.h"
#include "thrust/sort.h"

#define ERRORCHECK 1

//integrator
#define DIRECTLIGHTING 0
#define FULLLIGHTING 1
#define MISDIRECTLIGHTING 0


//optimize
#define SORTBYMATERIAL 0
#define CACHEFIRSTBOUNCE 1


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
static ShadeableIntersection* dev_firstIntersections = NULL;
static Geom* dev_lights = NULL;
static PathSegment* dev_paths2 = NULL;
static PathSegment* dev_paths3 = NULL;
static bool* dev_fullBools = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
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
	cudaMalloc(&dev_firstIntersections, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Geom));
	cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);

#if MISDIRECTLIGHTING || FULLLIGHTING
	cudaMalloc(&dev_paths2, pixelcount * sizeof(PathSegment));
#endif
#if FULLLIGHTING
	cudaMalloc(&dev_paths3, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_fullBools, pixelcount * sizeof(bool));
#endif
	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_firstIntersections);
	cudaFree(dev_lights);
#if MISDIRECTLIGHTING || FULLLIGHTING
	cudaFree(dev_paths2);
#endif
#if FULLLIGHTING
	cudaFree(dev_paths3);
	cudaFree(dev_fullBools);
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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(0.0f); 
		segment.beta = glm::vec3(1.f);
		segment.lightGeomId = -1;  //the light been chosen for this path

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
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
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; ++i)
		{
			//get every object in the scene, and do a intersection test for every one of them.
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SQUARE_PLANE) {
				t = squarePlaneIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
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
			intersections[path_index].materialId = INT_MAX;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].geomId = hit_geom_index;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

__global__ void updatePathSegByIntersections(
	PathSegment* pathSegments,
	ShadeableIntersection* intersections,
	const int num_paths,
	Material* materials,
	bool* bools,
	int depth
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_paths) {
		ShadeableIntersection& intersection = intersections[idx];
		PathSegment& pathSegment = pathSegments[idx];
		if (intersection.t <= 0.f) {
			pathSegment.remainingBounces = 0;
			bools[idx] = false;
		}
		else {
			Material& material = materials[intersection.materialId];
			if (material.emittance > 0.f) {
				if (depth == 0) {
					pathSegment.color = material.emittance * material.color;
				}
				pathSegment.remainingBounces = 0;
				bools[idx] = false;
			}
			else {
				bools[idx] = true;
			}
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
__global__ void shadeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, Geom* dev_geoms
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float pdf_f_f;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			// here we assume the ray stops when it hit anything with emittance bigger than 0.f
			if (material.emittance > 0.0f) {
				pathSegments[idx].color += pathSegments[idx].beta * (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0;
			}
			else {
				glm::vec3 intersect = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;
				Geom& intersectGeom = dev_geoms[intersection.geomId];
				scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng, pdf_f_f);
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
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
* Used for stream compaction
*/
struct isTerminate {
	__host__ __device__
		bool operator()(const ShadeableIntersection& intersection) {
		return intersection.t > -1.0;
	}
};

/**
* Used for material sort
*/
struct compareMaterialId {
	__host__ __device__
		bool operator()(const ShadeableIntersection& isectA, const ShadeableIntersection& isectB) {
		return isectA.materialId < isectB.materialId;
	}
};

struct remainingBounceIsNot0 {
	__host__ __device__
		bool operator()(const PathSegment& p1) {
		return (p1.remainingBounces > 0);
	}
};


void naiveIntegrator(int iter, 
	int pixelcount, 
	int traceDepth, 
	int blockSize1d) 
{
	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = pixelcount;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks(intersections)
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if CACHEFIRSTBOUNCE
		//iter starts with 1
		if (iter > 1 && depth == 0) {
			cudaMemcpy(dev_intersections, dev_firstIntersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				);
			cudaDeviceSynchronize();
			if (iter == 1 && depth == 0) {
				cudaMemcpy(dev_firstIntersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
		}
#else
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
#endif
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		++depth;

		// sort by material
#if SORTBYMATERIAL
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compareMaterialId());
#endif
		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_geoms
			);
		cudaDeviceSynchronize();

		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, remainingBounceIsNot0());
		num_paths = dev_path_end - dev_paths;

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
		if (num_paths == 0 || traceDepth == depth) {
			iterationComplete = true;
		}
	}
}


/**
* sample a direction, update beta
*/
__global__ void shadeMaterialDirectLight(
	const int iter
	, const int num_paths
	, const ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, const Material* materials
	, const Geom* dev_lights
	, const int num_lights
	, const int num_geoms
	, Geom* dev_geoms
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...

			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			glm::vec3 intersect = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;

			if (material.emittance > 0.0f) {
				pathSegments[idx].color += pathSegments[idx].beta * (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0;
			}
			else {
				//randomly choose a light source and a point on that light source to produce a ray
				//and update pathsegment's ray direction, origin and beta
				float pdf_l_l;
				float pdf_l_f;
				scatterRayToLight(
					pathSegments[idx],
					intersect,
					intersection.surfaceNormal,
					material,
					rng,
					pathSegments[idx].lightGeomId,
					dev_lights,
					num_lights,
					pdf_l_l,
					pdf_l_f
				);
				float pdf_null;
				int lightGeomId = pathSegments[idx].lightGeomId;
				bool outside_null;
				if (computeIntersectionWithLight(pathSegments[idx], dev_geoms, num_geoms, lightGeomId,
					intersect, intersection.surfaceNormal, outside_null, pdf_null, num_lights)) {
					Geom& light = dev_geoms[lightGeomId];
					float emittance = materials[light.materialid].emittance;
					pathSegments[idx].color += emittance * pathSegments[idx].beta;
				}
				else {
					pathSegments[idx].color = glm::vec3(0.f);
				}
				pathSegments[idx].remainingBounces = 0;
			}
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
		}
	}
}


__global__ void shadeMaterialMISLight(
	const int iter
	, const int num_paths
	, const ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, const Material* materials
	, const Geom* dev_lights
	, const int num_lights
	, Geom* dev_geoms
	, const int num_geoms
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...

			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			glm::vec3 intersect = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;

			if (material.emittance > 0.0f) {
				pathSegments[idx].color += materialColor * material.emittance;
				pathSegments[idx].remainingBounces = 0;
			}
			else {
				//randomly choose a light source and a point on that light source to produce a ray
				//and update pathsegment's ray direction, origin and beta
				float pdf_l_l;
				float pdf_l_f;
				scatterRayToLight(
					pathSegments[idx],
					intersect,
					intersection.surfaceNormal,
					material,
					rng,
					pathSegments[idx].lightGeomId,
					dev_lights,
					num_lights,
					pdf_l_l,
					pdf_l_f
				);
				float wg = powerHeuristic(1, pdf_l_l, 1, pdf_l_f);
				pathSegments[idx].beta *= wg;
				float pdf_null;
				int lightGeomId = pathSegments[idx].lightGeomId;
				if (computeIntersectionWithLight(pathSegments[idx], dev_geoms, num_geoms, lightGeomId,
					intersect, intersection.surfaceNormal, false, pdf_null, num_lights)) {
					Geom& light = dev_geoms[lightGeomId];
					float emittance = materials[light.materialid].emittance;
					pathSegments[idx].color += emittance * pathSegments[idx].beta;
				}
				else {
					pathSegments[idx].color = glm::vec3(0.f);
				}
			}
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
		}
	}
}

__global__ void shadeMaterialFullLight(
	const int iter
	, const int num_paths
	, const ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, PathSegment* pathSegments2
	, const Material* materials
	, const Geom* dev_lights
	, const int num_lights
	, Geom* dev_geoms
	, const int num_geoms
	, bool* bools
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		if (bools[idx]) {
			ShadeableIntersection intersection = shadeableIntersections[idx];

			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			int materialId = intersection.materialId;

			Material material = materials[materialId];
			glm::vec3 materialColor = material.color;

			glm::vec3 intersect = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;
			//randomly choose a light source and a point on that light source to produce a ray
			//and update pathsegment's ray direction, origin and beta
			float pdf_l_l;
			float pdf_l_f;
			scatterRayToLight(
				pathSegments2[idx],
				intersect,
				intersection.surfaceNormal,
				material,
				rng,
				pathSegments2[idx].lightGeomId,
				dev_lights,
				num_lights,
				pdf_l_l,
				pdf_l_f
			);
			float wg = powerHeuristic(1, pdf_l_l, 1, pdf_l_f);
			pathSegments2[idx].beta *= wg;
			float pdf_null;
			int lightGeomId = pathSegments2[idx].lightGeomId;
			if (computeIntersectionWithLight(pathSegments2[idx], dev_geoms, num_geoms, lightGeomId,
				intersect, intersection.surfaceNormal, false, pdf_null, num_lights)) {
				Geom& light = dev_geoms[lightGeomId];
				float emittance = materials[light.materialid].emittance;
				glm::vec3 materialColor = materials[light.materialid].color;
				pathSegments2[idx].color += emittance * pathSegments2[idx].beta * materialColor;
			}
			else {
				pathSegments2[idx].color = glm::vec3(0.f);
			}
		}
	}
}
/**
* for MIS bsdf ray, we make sure each pathSegment not hit a light!(因为之前partition过了)
*/
__global__ void shadeMaterialMIS(
	const int iter
	, const int num_paths
	, const ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, PathSegment* pathSegments2
	, Material* materials
	, int geomSize
	, int num_lights
	, Geom* dev_geoms
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) {
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;
			glm::vec3 intersect = pathSegments2[idx].ray.origin + intersection.t * pathSegments2[idx].ray.direction;
			
			scatterRayMIS(pathSegments[idx], pathSegments2[idx], intersect, intersection.surfaceNormal, material, rng, dev_geoms, num_lights, geomSize, materials);
		}
	}
}

__global__ void combineTwoSeg(
	PathSegment* pathSegments,
	PathSegment* pathSegments2,
	int num_paths
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		pathSegments[idx].color += pathSegments2[idx].color;
	}
}

__global__ void combineTwoSegFull (
	PathSegment* pathSegments,
	PathSegment* pathSegments2,
	int num_paths,
	bool* bools
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		pathSegments[idx].color += pathSegments2[idx].color;
	}
}

__global__ void shadeMaterialFullBSDF(
	const int iter,
	const int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	PathSegment* pathSegments2,
	PathSegment* pathSegments3,
	Material* materials,
	const int num_geoms,
	const int num_lights,
	Geom* dev_geoms,
	bool* bools
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		if (bools[idx]) {
			ShadeableIntersection intersection = shadeableIntersections[idx];
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			Material material = materials[intersection.materialId];
			glm::vec3 intersect = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;
			scatterRayMIS(pathSegments2[idx], pathSegments3[idx], intersect, intersection.surfaceNormal, material, rng, dev_geoms, num_lights, num_geoms, materials);
		}
	}
}

__global__ void shadeMaterialFullMainPath(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, PathSegment* pathSegments2
	, bool* bools
	, int depth
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		if (bools[idx]) {
			ShadeableIntersection intersection = shadeableIntersections[idx];
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;
			PathSegment& pathSegment = pathSegments[idx];
			glm::vec3 intersect = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
			float pdf_f_f;
			//combine color
			pathSegment.color += pathSegment.beta * pathSegments2[idx].color;
			//update完color, 再generate新的ray
			scatterRayFullLight(pathSegment, intersect, intersection.surfaceNormal, material, rng, pdf_f_f);
			if (depth > 3) {
				//when rayColor's max rgb is smaller, q is bigger, more likely to terminate
				float q = glm::max(0.05f, 1.f - glm::max(glm::max(pathSegment.beta.x, pathSegment.beta.y), pathSegment.beta.z));
				pathSegment.beta /= (1 - q);
				thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
				thrust::uniform_real_distribution<float> u01(0, 1);
				if (u01(rng) < q) {
					pathSegment.remainingBounces = 0;
				}
			}
		}
	}
}

__global__ void resetPaths(
	PathSegment* dev_paths2,
	PathSegment* dev_paths3,
	const int num_paths) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < num_paths)
	{
		dev_paths2[index].lightGeomId = -1;
		dev_paths2[index].color = glm::vec3(0.f);
		dev_paths2[index].beta = glm::vec3(1.f);
		dev_paths2[index].remainingBounces = 1;
		dev_paths3[index].lightGeomId = -1;
		dev_paths3[index].color = glm::vec3(0.f);
		dev_paths3[index].beta = glm::vec3(1.f);
		dev_paths3[index].remainingBounces = 1;
	}
}

void directLightIntegrator(int iter, int pixelcount, int blockSize1d, int num_lights, int num_geoms) {
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = pixelcount;
	
	//depth 0
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
	if (iter > 1) {
		cudaMemcpy(dev_intersections, dev_firstIntersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
	}
	else {
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
			cudaMemcpy(dev_firstIntersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
	}
#if SORTBYMATERIAL
	thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compareMaterialId());
#endif
	//shade and choose a vec from intersect to light
	shadeMaterialDirectLight<<<numblocksPathSegmentTracing, blockSize1d >>>(
		iter
		, num_paths
		, dev_intersections
		, dev_paths
		, dev_materials
		, dev_lights
		, num_lights
		, num_geoms
		, dev_geoms
	);

	if (guiData != NULL)
	{
		guiData->TracedDepth = 2;
	}
}

void misDirLightIntegrator(int iter, int pixelcount, int blockSize1d, int num_lights, int num_geoms) {
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = pixelcount;

	//depth 0
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
	if (iter > 1) {
		cudaMemcpy(dev_intersections, dev_firstIntersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
	}
	else {
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
		cudaMemcpy(dev_firstIntersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
	}

	//depth 1
	numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
	if (guiData != NULL)
	{
		guiData->TracedDepth = 1;
	}

	//give the end result color for l
	shadeMaterialMISLight <<<numblocksPathSegmentTracing, blockSize1d >>> (
		iter
		, num_paths
		, dev_intersections
		, dev_paths
		, dev_materials
		, dev_lights
		, num_lights
		, dev_geoms
		, num_geoms
	);

	//sort once, clean those paths that hit a light or hit nothing
	dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, dev_paths2, remainingBounceIsNot0());
	num_paths = dev_path_end - dev_paths;

	//shade another ray by bsdf, update its beta, with wf
	shadeMaterialMIS << <numblocksPathSegmentTracing, blockSize1d >> > (
		iter,
		num_paths,
		dev_intersections,
		dev_paths,
		dev_paths2,
		dev_materials,
		hst_scene->geoms.size(),
		num_lights,
		dev_geoms
		);

	//combine the two pathSegment
	combineTwoSeg <<<numblocksPathSegmentTracing, blockSize1d >>>(
		dev_paths, 
		dev_paths2, 
		pixelcount
		);

	//depth 2
	if (guiData != NULL)
	{
		guiData->TracedDepth = 2;
	}
}

void fullLightIntegrator(int iter, int pixelcount, int traceDepth, int blockSize1d, int num_lights, int num_geoms) {

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = pixelcount;

	dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

	bool iterationComplete = false;
	while (!iterationComplete) {
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		resetPaths << <numblocksPathSegmentTracing, blockSize1d >> > (dev_paths2, dev_paths3, num_paths);
		numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

		if (iter > 1 && depth == 0) {
			cudaMemcpy(dev_intersections, dev_firstIntersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				);
			cudaDeviceSynchronize();
			if (iter == 1 && depth == 0) {
				cudaMemcpy(dev_firstIntersections, dev_intersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
		}
		cudaDeviceSynchronize();
		
		updatePathSegByIntersections <<<numblocksPathSegmentTracing, blockSize1d >>> (dev_paths, dev_intersections, num_paths, dev_materials, dev_fullBools, depth);
		cudaDeviceSynchronize();

		//light
		shadeMaterialFullLight << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter
			, num_paths
			, dev_intersections
			, dev_paths
			, dev_paths2
			, dev_materials
			, dev_lights
			, num_lights
			, dev_geoms
			, num_geoms
			, dev_fullBools
			);
		cudaDeviceSynchronize();
		
		//bsdf
		shadeMaterialFullBSDF << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_paths2,
			dev_paths3,
			dev_materials,
			hst_scene->geoms.size(),
			num_lights,
			dev_geoms,
			dev_fullBools
			);
		cudaDeviceSynchronize();

		combineTwoSegFull << <numblocksPathSegmentTracing, blockSize1d >> > (
			dev_paths2,
			dev_paths3,
			num_paths,
			dev_fullBools
			);
		cudaDeviceSynchronize();

		shadeMaterialFullMainPath << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_paths2,
			dev_fullBools,
			depth
			);
		cudaDeviceSynchronize();

		++depth;

		//sort
		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, remainingBounceIsNot0());
		num_paths = dev_path_end - dev_paths;

		if (num_paths == 0 || traceDepth == depth) {
			break;
		}

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

}


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 *
 * @param iter: number of interation from runCuda()
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;   //the depth of each ray
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;
	const int num_lights = hst_scene->lights.size();
	const int num_geoms = hst_scene->geoms.size();

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);   //64 threads per block
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

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

#if DIRECTLIGHTING
	directLightIntegrator(iter, pixelcount, blockSize1d, num_lights, num_geoms);
#elif MISDIRECTLIGHTING
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths2);
	misDirLightIntegrator(iter, pixelcount, blockSize1d, num_lights, num_geoms);
#elif FULLLIGHTING
	fullLightIntegrator(iter, pixelcount, traceDepth, blockSize1d, num_lights, num_geoms);
#else
	naiveIntegrator(iter, pixelcount, traceDepth, blockSize1d);
#endif


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
}