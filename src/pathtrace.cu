#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define SORT_BY_MATERIAL 0
#define CACHE_FIRST_INTERSECTION 0

#define ANTIALIASING 0

#define MOTION_BLUR 0
#define MOTION_VELOCITY glm::vec3(0.0f, 0.75f, 0.0f)

#define DEPTH_OF_FIELD 1
#define LENS_RADIUS 4.0f
#define FOCAL_DISTANCE 4.f
#define PI 3.141592654f

#define DIRECT_LIGHTING 0

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

struct pathTerminated
{
	__host__ __device__
		bool operator()(const PathSegment& pathSegment)
	{
		return pathSegment.remainingBounces == 0;
	}
};

struct pathTerminatedPartition
{
	__host__ __device__
		bool operator()(const PathSegment& pathSegment)
	{
		return pathSegment.remainingBounces != 0;
	}
};

struct sortByMaterial
{
	__host__ __device__
		bool operator()(const ShadeableIntersection & intersection1, const ShadeableIntersection & intersection2) {
		return intersection1.materialId > intersection2.materialId;
	}
};

// https://sci-hub.se/10.1080/10867651.1997.10487479
__host__ __device__
glm::vec3 squareToDiskConcentric(const glm::vec2& sample)
{
	float a = 2.f * sample.x - 1.f;
	float b = 2.f * sample.y - 1.f;
	float r;
	float theta;
	float pi = PI / 4.f;
	if (a > -b) {
		if (a > b) {
			r = a;
			theta = pi * (b / a);
		}
		else {
			r = b;
			theta = pi * (2.f - (a / b));
		}
	}
	else
	{
		if (a < b) {
			r = -a;
			theta = pi * (4.f + (b / a));
		}
		else {
			r = -b;
			if (b != 0) {
				theta = pi * (6.f - (a / b));
			}
			else {
				theta = 0.f;
			}
		}
	}
	float x = r * cos(theta);
	float y = r * sin(theta);
	return glm::vec3(x, y, 0.f);
}

__device__ __host__ 
glm::vec3 pointOnSquarePlane(thrust::default_random_engine& rng, Geom light)
{
	thrust::uniform_real_distribution<float> u01(0, 1);
	glm::vec2 randpoint(u01(rng), u01(rng));
	glm::vec3 pointOnPlane = glm::vec3((randpoint - glm::vec2(0.5f)), 0.f);
	//from -0.5, 0.5 to world space
	glm::vec3 pointOnPlaneWorld = glm::vec3(light.transform * glm::vec4(pointOnPlane, 1.f));
	return pointOnPlaneWorld;
}

__host__ __device__
float heartFunction(const glm::vec2& sample) {
	float tmp = (sample.x * sample.x + sample.y * sample.y - 1.0f);
	return (tmp * tmp * tmp - sample.x * sample.x * sample.y * sample.y * sample.y);
}
__host__ __device__
glm::vec3 squareToHeart(const glm::vec2& sample)
{
	float isHeart = heartFunction(sample);
	if (isHeart < 0.0f) {
		return glm::vec3(sample.x, sample.y, 0.f);
	}
	return glm::vec3(0.f, 0.f, 0.f);
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
static PathSegment* dev_first_paths = NULL;
static ShadeableIntersection* dev_first_intersections = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static Triangle* dev_triangles = NULL;

#if DIRECT_LIGHTING
static Geom* dev_lights = NULL;
#endif

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
	cudaMalloc(&dev_first_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	cudaMalloc(&dev_first_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_first_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need
	// if geoms contain mesh, allocate dev_triangles for it
	if (scene->hasMesh && scene->meshGeomId != -1) {
		Geom mesh = scene->geoms[scene->meshGeomId];
		cudaMalloc(&dev_triangles, mesh.numTris * sizeof(Triangle));
		cudaMemcpy(dev_triangles, mesh.triangles, mesh.numTris * sizeof(Triangle), cudaMemcpyHostToDevice);
	}

#if DIRECT_LIGHTING
	cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Geom));
	cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);
#endif

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_first_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	cudaFree(dev_first_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_triangles);
#if DIRECT_LIGHTING
	cudaFree(dev_lights);
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

		// calculate ray origin
#if DEPTH_OF_FIELD
		// random a sample point on the disk as a point on lens
		// added it to the camera origin
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
		/*thrust::uniform_real_distribution<float> u01(0, 1);
		glm::vec3 pointOnLens = squareToDiskConcentric(glm::vec2(u01(rng), u01(rng))) * LENS_RADIUS;*/
		thrust::uniform_real_distribution<float> u01(-1.4, 1.4);
		glm::vec3 pointOnLens = squareToHeart(glm::vec2(u01(rng), u01(rng))) * LENS_RADIUS;
		glm::vec3 origin =  cam.position + glm::mat3(cam.right, cam.up, cam.view) * pointOnLens;
		segment.ray.origin = origin;
#else
		segment.ray.origin = cam.position;
#endif

		// TODO: implement antialiasing by jittering the ray
		// if antialiasing, pointing the ray to different position around the original reference point 
		// to blur the result

		// calculate ray direction
#if ANTIALIASING
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
		thrust::uniform_real_distribution<float> u01(-0.5, 0.5);
		float xOffset = u01(rng);
		float yOffset = u01(rng);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + xOffset)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + yOffset)
		);
#elif DEPTH_OF_FIELD
		// calculate new ref point
		float ndc_x = 1.f - ((float)x / cam.resolution.x) * 2.f;
		float ndc_y = 1.f - ((float)y / cam.resolution.y) * 2.f;

		glm::vec3 ref = cam.position + FOCAL_DISTANCE * cam.view;

		float angle = glm::radians(cam.fov.y);
		float tan_fovy = tan(angle);
		float len = glm::length(ref - cam.position);

		float aspect = (float)cam.resolution.x / (float)cam.resolution.y;
		glm::vec3 V = cam.up * len * tan_fovy;
		glm::vec3 H = cam.right * len * aspect * tan_fovy;

		glm::vec3 P = ref + ndc_x * H + ndc_y * V;
		glm::vec3 direction = glm::normalize(P - segment.ray.origin);
		segment.ray.direction = direction;
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
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
	, int geoms_size
	, ShadeableIntersection* intersections
	, Triangle* triangles,
	int iter
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

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
#if MOTION_BLUR
				thrust::default_random_engine rng = makeSeededRandomEngine(iter, path_index, 0);
				thrust::uniform_real_distribution<float> u01(0, 1);
				Ray motionBlurRay = pathSegment.ray;
				glm::vec3 motionBlurRayOrigin = u01(rng) * MOTION_VELOCITY;
				motionBlurRay.origin += motionBlurRayOrigin;
				t = sphereIntersectionTest(geom, motionBlurRay, tmp_intersect, tmp_normal, outside);
#else
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
#endif
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?
			else if (geom.type == MESH) {
				// ray test intersection with every triangle in mesh
				t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, triangles, outside);
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
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
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
				pathSegments[idx].color *= (materialColor * material.emittance);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				pathSegments[idx].color *= u01(rng); // apply some noise because why not
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
		}
	}
}

__global__ void shadeBaseOnMaterial(
	int iter, 
	int num_paths, 
	ShadeableIntersection* shadeableIntersections, 
	PathSegment* pathSegments, 
	Material* materials) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		PathSegment& pathSegment = pathSegments[idx];
		ShadeableIntersection intersection = shadeableIntersections[idx];

		// if intersection
		if (pathSegment.remainingBounces > 0 && intersection.t > 0.0f) {
			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			// if hit light
			if (material.emittance > 0.0f) {
				// multiple the material color with ray color
				pathSegment.color *= (materialColor * material.emittance);
				// terminate the ray
				pathSegment.remainingBounces = 0;
			}
			else {
				//sample the material BSDFs to generate a new ray, update the pathSegment with the new ray
				glm::vec3 intersectionPoint = getPointOnRay(pathSegment.ray, intersection.t);
				scatterRay(pathSegment, intersectionPoint, intersection.surfaceNormal, material, rng);
				pathSegment.remainingBounces --;
			}
		}
		// if not intersection
		else {
			// terminate ray
			pathSegment.remainingBounces = 0;
			// color black
			pathSegment.color = glm::vec3(0.f, 0.f, 0.f);
		}
	}
}

__global__ void shadeBaseOnMaterialDirectLighting(
	int iter,
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials,
	Geom* lights,
	int numLights)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		PathSegment& pathSegment = pathSegments[idx];
		ShadeableIntersection intersection = shadeableIntersections[idx];
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);

		// if intersection and is not the last two boundce
		// do as usual
		if (pathSegment.remainingBounces != 2 && pathSegment.remainingBounces > 0 && intersection.t > 0.0f) {
			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;
			// if hit light
			if (material.emittance > 0.0f) {
				// multiple the material color with ray color
				pathSegment.color *= (materialColor * material.emittance);
				// terminate the ray
				pathSegment.remainingBounces = 0;
			}
			else if (material.emittance > 0.0f && pathSegment.remainingBounces == 1) {
				pathSegment.color *= (materialColor * material.emittance * (float)numLights);
			}
			else {
				//sample the material BSDFs to generate a new ray, update the pathSegment with the new ray
				glm::vec3 intersectionPoint = getPointOnRay(pathSegment.ray, intersection.t);
				scatterRay(pathSegment, intersectionPoint, intersection.surfaceNormal, material, rng);
				pathSegment.remainingBounces--;
			}
		}
		// if is the bounce before the last bounce
		// randomly select a light
		// randomly select a point on it
		// scatter ray to that point
		else if (pathSegment.remainingBounces == 2 && intersection.t > 0.0f) {
			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// if hit light
			if (material.emittance > 0.0f) {
				// multiple the material color with ray color
				pathSegment.color *= (materialColor * material.emittance);
				// terminate the ray
				pathSegment.remainingBounces = 0;
			}
			else {
				//sample the material BSDFs to generate a new ray, update the pathSegment with the new ray
				glm::vec3 intersectionPoint = getPointOnRay(pathSegment.ray, intersection.t);
				scatterRay(pathSegment, intersectionPoint, intersection.surfaceNormal, material, rng);
				// select a light randomly
				thrust::uniform_real_distribution<float> u01(0, 1);
				float rand = u01(rng);
				int lightNum = glm::min((int)(rand * numLights), numLights - 1);
				Geom light = lights[lightNum];
				glm::vec3 pointOnLight = pointOnSquarePlane(rng, light);
				glm::vec3 newRayDir = glm::normalize(pointOnLight - pathSegment.ray.origin);
				pathSegment.ray.direction = newRayDir;
				pathSegment.remainingBounces--;
			}
		}
		// if not intersection
		else {
			// terminate ray
			pathSegment.remainingBounces = 0;
			// color black
			pathSegment.color = glm::vec3(0.f, 0.f, 0.f);
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
#if CACHE_FIRST_INTERSECTION
	if (iter == 1) {
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
		checkCUDAError("generate camera ray");
		cudaMemcpy(dev_first_paths, dev_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	}
	else {
		cudaMemcpy(dev_paths, dev_first_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	}
#else
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
#endif


	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;


		// if is the first iteration and depth is 0, compute the intersections and cache them
		if (CACHE_FIRST_INTERSECTION && iter == 1 && depth == 0 ) {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, dev_triangles,
				iter
				);
			cudaMemcpy(dev_first_intersections, dev_intersections,  pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			checkCUDAError("trace one bounce");
		}
		// if is not the first iteration and depth is 0, copy the first intersections
		else if (CACHE_FIRST_INTERSECTION && iter != 1 && depth == 0){
			cudaMemcpy(dev_intersections, dev_first_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		// if is not first intersection and depth is not 0 calculate intersections as usual
		else {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, dev_triangles,
				iter
				);
			checkCUDAError("trace one bounce");
		}
		cudaDeviceSynchronize();
		depth++;

#if SORT_BY_MATERIAL
		// sort by material
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, sortByMaterial());
#endif

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.
#if DIRECT_LIGHTING
		shadeBaseOnMaterialDirectLighting << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_lights,
			hst_scene->lights.size()
			);
		checkCUDAError("shade base on material direct lighting");
#else
		shadeBaseOnMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);
		checkCUDAError("shade base on material");
#endif
		// stream compaction to remove the terminated pathSegment
		//dev_path_end = thrust::remove_if(thrust::device, dev_paths, dev_path_end, pathTerminated());
		//num_paths = dev_path_end - dev_paths;
		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, pathTerminatedPartition());
		num_paths = dev_path_end - dev_paths;
		//std::cout << "num_paths: " << num_paths << std::endl;
		if (num_paths == 0 || depth == traceDepth) {
			iterationComplete = true; // TODO: should be based off stream compaction results.
			//std::cout << "iteration complete!" << std::endl;
		}


		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
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
}
