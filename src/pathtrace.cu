#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/partition.h>


#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include "../stream_compaction/efficient.h"


#define ERRORCHECK 0
#define SORT_RAYS 0
#define CACHE_FIRST_BOUNCE 0
#define ANTI_ALIASING 0

#define DEPTH_OF_FIELD 0
#define FOCAL_LENGTH 10.0f
#define APERTURE 0.3f

#define MESH_BOUNDING_BOX 1

#define POST_PROCESS 0
#define GREYSCALE 0
#define SEPIA 0
#define INVERTED 0
#define CONTRAST 1

#define MOTION_BLUR 0

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
// ...
static ShadeableIntersection* dev_intersection_cache = NULL;
static Triangle* dev_triangles = NULL;
static Texture* dev_textures = NULL;
static glm::vec3* dev_texColors = NULL;

static Geom* dev_lights = NULL;



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
	cudaMalloc(&dev_intersection_cache, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersection_cache, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
	cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_texColors, scene->textureColors.size() * sizeof(glm::vec3));
	cudaMemcpy(dev_texColors, scene->textureColors.data(), scene->textureColors.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Geom));
	cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_intersection_cache);
	cudaFree(dev_triangles);
	cudaFree(dev_textures);
	cudaFree(dev_texColors);
	cudaFree(dev_lights);

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
	int index = x + (y * cam.resolution.x);

	thrust::default_random_engine rng = makeSeededRandomEngine(iter, index , pathSegments[index].remainingBounces);
	thrust::uniform_real_distribution<float> u01(-0.5, 0.5);


	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);



		// TODO: implement antialiasing by jittering the ray
#if ANTI_ALIASING
		float jX = u01(rng);
		float jY = u01(rng);
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + jX - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + jY - (float)cam.resolution.y * 0.5f)
		);
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif

#if DEPTH_OF_FIELD
		float jXd = u01(rng);
		float jYd = u01(rng);

		glm::vec3 focalPoint = segment.ray.direction * FOCAL_LENGTH;
		glm::vec3 shift = glm::vec3(jXd, jYd, 0.0f) * APERTURE;

		segment.ray.origin += shift;
		segment.ray.direction = glm::normalize(focalPoint - shift);
#endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}


__host__ __device__ bool checkMeshhBoundingBox(Geom& geom, Ray& ray) {
	Ray q;
	q.origin = multiplyMV(geom.inverseTransform, glm::vec4(ray.origin, 1.0f));
	q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(ray.direction, 0.0f)));

	float tmin = -1e38f;
	float tmax = 1e38f;
	glm::vec3 tmin_n;
	glm::vec3 tmax_n;
	for (int xyz = 0; xyz < 3; ++xyz) {
		float qdxyz = q.direction[xyz];
		if (glm::abs(qdxyz) > 0.00001f) {
			float t1 = (geom.boundingBoxMin[xyz] - q.origin[xyz]) / qdxyz;
			float t2 = (geom.boundingBoxMax[xyz] - q.origin[xyz]) / qdxyz;
			float ta = glm::min(t1, t2);
			float tb = glm::max(t1, t2);
			glm::vec3 n;
			n[xyz] = t2 < t1 ? +1 : -1;
			if (ta > 0 && ta > tmin) {
				tmin = ta;
				tmin_n = n;
			}
			if (tb < tmax) {
				tmax = tb;
				tmax_n = n;
			}
		}
	}

	if (tmax >= tmin && tmax > 0) {
		return true;
	}
	return false;
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
	// for mesh loading
	,Triangle* triangles
	,int iter
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

		glm::vec2 uv(0.f);

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
			// TODO: add more intersection tests here... triangle? metaball? CSG?
			else if (geom.type == OBJ_MESH)
			{
#if MESH_BOUNDING_BOX
				if (checkMeshhBoundingBox(geom, pathSegment.ray))
				{
					t = triangleIntersectionTest(geom, pathSegment.ray,
						tmp_intersect, triangles, geom.triangleStart, geom.triangleEnd, tmp_normal, outside, uv);
				}
#else
				t = triangleIntersectionTest(geom, pathSegment.ray,
					tmp_intersect, triangles + geom.triangleIndex, triangles_size, tmp_normal, outside, uv);
#endif
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
			// Suggested by Rhuta
			pathSegments[path_index].remainingBounces = 0;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].uv = uv;
			intersections[path_index].textureId = geoms[hit_geom_index].textureId;
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

__global__ void shadeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, Texture* textures
	, glm::vec3* texColors
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths && pathSegments[idx].remainingBounces >= 0)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		// No intersection then return black and no more bounce
		if (intersection.t <= 0.0f)
		{
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
			return;
		}
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);

		Material material = materials[intersection.materialId];
		glm::vec3 materialColor = material.color;

		if (intersection.textureId != -1)
		{
			Texture tex = textures[intersection.textureId];
			int w = tex.width * intersection.uv[0] - 0.5;
			int h = tex.height * (1 - intersection.uv[1]) - 0.5;
			int colIdx = h * tex.width + w + tex.idx;
			material.color = texColors[colIdx];
		}

		// Light source then return light color 
		if (material.emittance > 0.0f)
		{
			pathSegments[idx].color *= (materialColor * material.emittance);
			pathSegments[idx].remainingBounces = 0;
			return;
		}
		// ScatterRay and accumulate color
		else
		{
			glm::vec3 i = getPointOnRay(pathSegments[idx].ray, intersection.t);


			scatterRay(pathSegments[idx], i, intersection.surfaceNormal, material, rng);
			return;
		}
	}
}

__global__ void shadeMaterialWithDirectLighting(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, Geom* lights
	, int lightCount
	,Texture* textures
    ,glm::vec3* texColors
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths && pathSegments[idx].remainingBounces >= 0)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		// No intersection then return black and no more bounce
		if (intersection.t <= 0.0f)
		{
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
			return;
		}
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);

		Material material = materials[intersection.materialId];
		glm::vec3 materialColor = material.color;

		glm::vec3 textureColor = glm::vec3(0.f);
		if (intersection.textureId != -1)
		{
			Texture tex = textures[intersection.textureId];
			int w = tex.width * intersection.uv[0] - 0.5;
			int h = tex.height * (1 - intersection.uv[1]) - 0.5;
			int colIdx = h * tex.width + w;
			material.color = texColors[colIdx];
		}


		// Light source then return light color 
		if (material.emittance > 0.0f)
		{
			pathSegments[idx].color *= (materialColor * material.emittance);
			pathSegments[idx].remainingBounces = 0;
			return;
		}
		// ScatterRay and accumulate color
		else
		{
			glm::vec3 i = getPointOnRay(pathSegments[idx].ray, intersection.t);

			scatterRay(pathSegments[idx], i, intersection.surfaceNormal, material, rng);
			
			if (pathSegments[idx].remainingBounces == 1)
			{
				thrust::uniform_real_distribution<float> u01(0, 1);
				thrust::uniform_real_distribution<int> u02(0, lightCount-1);
				glm::vec3 sampledLight = glm::vec3(lights[u02(rng)].transform * glm::vec4(u01(rng), u01(rng), u01(rng), 1.f));
				pathSegments[idx].ray.direction = glm::normalize(sampledLight - pathSegments[idx].ray.origin);
			}

			return;
		}
	}
}
__global__ void postShade(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		glm::vec3 postProcessColor = glm::vec3(0.f);

		glm::vec3 currentColor = pathSegments[idx].color;

		// Greyscale filter
#if GREYSCALE
		postProcessColor = glm::vec3(0.21 * currentColor.x + 0.72 * currentColor.y + 0.07 * currentColor.z);
#elif SEPIA
		// Sepia
		float adjust = 0.4f;
		postProcessColor.r = glm::min(1.0, (currentColor.r * (1.0 - (0.607 * adjust))) + (currentColor.g * (0.769 * adjust)) + (currentColor.b * (0.189 * adjust)));
		postProcessColor.g = glm::min(1.0, (currentColor.r * (0.349 * adjust)) + (currentColor.g * (1.0 - (0.314 * adjust))) + (currentColor.b * (0.168 * adjust)));
		postProcessColor.b = glm::min(1.0, (currentColor.r * (0.272 * adjust)) + (currentColor.g * (0.534 * adjust)) + (currentColor.b * (1.0 - (0.869 * adjust))));
#elif INVERTED 
		// Inverted
		postProcessColor = glm::vec3(1.0) - currentColor;
		// High Contrast
#elif CONTRAST
		postProcessColor =  (currentColor - glm::vec3(0.5f)) * 1.1f + glm::vec3(0.5f);

#endif

		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) {
			pathSegments[idx].color = postProcessColor;
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

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	int ori_num_paths = num_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

#if MOTION_BLUR
	for (int i = 0; i < hst_scene->geoms.size(); i++) {
		Geom& geom = hst_scene->geoms[i];
		geom.translation = geom.translation + (geom.endpos - geom.translation) * (float)iter / (float)hst_scene->state.iterations;
		geom.transform[3] = glm::vec4(geom.translation, geom.transform[3][3]);
		geom.inverseTransform = glm::inverse(geom.transform);
		geom.invTranspose = glm::inverseTranspose(geom.transform);
	}
	cudaMemcpy(dev_geoms, hst_scene->geoms.data(), hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
#endif

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));


		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

		
		// Caching the first intersection
#if CACHE_FIRST_BOUNCE
		if (depth == 0 && iter == 1)
		{
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, dev_triangles
				,iter
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
			cudaMemcpy(dev_intersection_cache, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else if (depth == 0)
		{
			cudaMemcpy(dev_intersections, dev_intersection_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else
		{
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, dev_triangles
				, iter
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}
#else
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			, dev_triangles
			, iter
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
#endif
		depth++;


		//Sort rays by material
		#if SORT_RAY
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compareIntersections())
#endif


			// TODO:
			// --- Shading Stage ---
			// Shade path segments based on intersections and generate new rays by
		  // evaluating the BSDF.
		  // Start off with just a big kernel that handles all the different
		  // materials you have in the scenefile.
		  // TODO: compare between directly shading the path segments and shading
		  // path segments that have been reshuffled to be contiguous in memory.

			//shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			//	iter,
			//	num_paths,
			//	dev_intersections,
			//	dev_paths,
			//	dev_materials
			//	);

#if DIRECT_LIGHTING
			shadeMaterialWithDirectLighting << <numblocksPathSegmentTracing, blockSize1d >> > (
				iter,
				num_paths,
				dev_intersections,
				dev_paths,
				dev_materials,
				dev_lights,
				hst_scene->lightCount,
				dev_textures,
				dev_texColors
					);
#else
		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_textures,
			dev_texColors
			);
#endif

#if POST_PROCESS

			postShade << <numblocksPathSegmentTracing, blockSize1d >> > (
				iter,
				num_paths,
				dev_intersections,
				dev_paths
				);
#endif

		//Stream compact
		PathSegment* path_end = thrust::stable_partition(thrust::device, dev_paths, dev_paths + num_paths, rayTerminated());
		num_paths = path_end - dev_paths;


		iterationComplete = (num_paths == 0); 


		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (ori_num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
