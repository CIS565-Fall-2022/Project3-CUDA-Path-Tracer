#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtc/epsilon.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

using namespace scene_structs;

#define ERRORCHECK 1
#define SORT_BY_MATERIALS 1
#define CACHE_FIRST_BOUNCE 1

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
		// Take the average for the monte carlo estimate over here!!!
		// Also you shouldn't really do average like this, however
		// The max value is only 255 so integer overflow is not much of a problem
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		//float cx = pix.x;
		//float cy = pix.y;
		//float cz = pix.z;
		//pix = glm::vec3(cx, cy, cz);

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
static int* dev_intersection_materials = NULL;
static int* dev_first_bounce_paths = NULL;
// same as scene_structs::Image, except pixels' data type is pointer to gpu buffer instead of vector
// instead of cpu vector
struct DevImage {
	int height;
	int width;
	int imageBufferOffset;
};
static DevImage *dev_imageSources;
static glm::vec3* dev_imageBuffers;

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
#if SORT_BY_MATERIALS
	cudaMalloc(&dev_intersection_materials, pixelcount * sizeof(int));
#endif

#if CACHE_FIRST_BOUNCE
	cudaMalloc(&dev_first_bounce_paths, pixelcount * sizeof(PathSegment));
#endif

	int currentOffset = 0;
	std::vector<DevImage> tempImages;
	cudaMalloc(&dev_imageSources, scene->images.size() * sizeof(DevImage));

	for (const auto &image: scene->images) {
		int imageSize = image.height * image.width;

		DevImage temp_image;
		temp_image.height = image.height;
		temp_image.width = image.width;
		temp_image.imageBufferOffset = currentOffset;
		tempImages.push_back(temp_image);

		currentOffset += imageSize;
	}

	cudaMemcpy(dev_imageSources, tempImages.data(), tempImages.size() * sizeof(DevImage), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy of dev_imageSources");

	// current offset = total number of pixels of all images
	cudaMalloc(&dev_imageBuffers, currentOffset * sizeof(glm::vec3));
	std::vector<glm::vec3> tempImageBuffers;

	for (const auto& image : scene->images) {
		tempImageBuffers.insert(tempImageBuffers.end(), image.pixels.begin(), image.pixels.end());
	}

	cudaMemcpy(dev_imageBuffers, tempImageBuffers.data(), tempImageBuffers.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy of dev_imageBuffers");

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
#if SORT_BY_MATERIALS
	cudaFree(dev_intersection_materials);
#endif
#if CACHE_FIRST_BOUNCE
	cudaFree(dev_first_bounce_paths);
#endif

	cudaFree(dev_imageSources);
	cudaFree(dev_imageBuffers);

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
		// dev_paths and dev_image are basically parallel arrays
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray

		// Without jittering, we wouldn't have anti-aliasing on the 1st ray bounce?
		// Since it would just point straight in a specific direction?
		// Could still get illumination from 2nd ray bounce?

		// eg. x index is (0, 1080) y index is (0, 720)
		// ((float)x - (float)cam.resolution.x * 0.5f) => basically convert this range to (-540, 540)
		// GUESSING: pixelLength.x is how many units correspond to 1 pixel in the camera?
		// So pixelLength.x here = 2 / resolution.x <- kinda. Also need to look at fov

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
// For each index in parallel, compute intersections of ray corresponding to index
// Intersection data has normal, distance raymarched (t), and material id
__global__ void computeIntersections(
	int depth
	, int num_paths
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

		float t; // GUESSING the distance marched by the ray
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1; // what object this intersection hit. Index should be index in dev_geoms
		bool outside = true; // if it hit outer surface of object or not. Not sure what to do if false

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv;

		// naive parse through global geoms
		// TODO*: replace this with kd tree or other acceleration structure...

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
			else if (geom.type == TRIANGLE) { // TODO: add more intersection tests here... triangle? metaball? CSG?

				t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside);
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
			intersections[path_index].t = -1.0f; // GUESSING: -1 means did not hit (raymarch didn't go anywhere?)
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

__device__ glm::vec3 getTextureColor(const DevImage& image, glm::vec3 *imageBuffers, const glm::vec2 &uv) {
	int h = (int) glm::floor(uv[0] * image.height);
	int w = (int) glm::floor(uv[1] * image.width);
	int index = image.imageBufferOffset + (h * image.height + w);
	return imageBuffers[index];
}

__global__ void shadeMaterial(
	int iter
	, int num_paths
	, int depth
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, DevImage* imageSources
	, glm::vec3* imageBuffers
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		PathSegment &pathSegment = pathSegments[idx];

		// If we didn't hit anything
		if (pathSegment.remainingBounces <= 0) {
			pathSegment.color = glm::vec3(0.0f);
			return;
		}

		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);

			Material &material = materials[intersection.materialId];
			glm::vec3 materialColor;

			if (material.colorImageId == -1) {
				materialColor = material.color;
			}
			else {
				DevImage& baseColorImage = imageSources[material.colorImageId];
				materialColor = getTextureColor(baseColorImage, imageBuffers, intersection.uv);
			}

			glm::vec3 intersectionPoint = intersection.t * pathSegment.ray.direction + pathSegment.ray.origin;
			scatterRay(pathSegment, intersectionPoint, intersection.surfaceNormal, material, rng);

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegment.color *= (materialColor * material.emittance);
				pathSegment.remainingBounces = 0; // hit light = stop tracing the path
				return;
			}
			else {
				// light is pointing from above?
				//float lightTerm = glm::clamp(glm::dot(intersection.surfaceNormal, newDirection), 0.0f, 1.0f);
				//pathSegments[idx].color *= material.color * lightTerm / pdf;

				// TOASK: this looks like the reference image, but isn't the reference image too dark?
				// We're doing cosine weighted sampling, so cosine in the denominator of the pdf
				// cancels with cosine in the numerator of the product
				// But isn't the pdf supposed to be cos(theta)/pi? shouldn't we multiply pi here?
				// Also, what happens for reflective materials?
				// Never mind, the answer has to be correct because it converges to a result
				// If we multiply PI the image gets brighter and brighter...
				pathSegment.color *= materialColor;

				//if (depth > 1) {
				//	float r = pathSegments[idx].color.r;
				//	float g = pathSegments[idx].color.g;
				//	float b = pathSegments[idx].color.b;
				//	float mr = material.color.r;
				//	float mg = material.color.g;
				//	float mb = material.color.b;
				//	material.color = glm::vec3(mr, mg, mb);
				//	pathSegments[idx].color = glm::vec3(r, g, b);
				//}

				//pathSegments[idx].color = iter % 2 == 0 ? glm::vec3(1, 0, 0) : glm::vec3(0, 0, 1);
				pathSegment.remainingBounces--;
			}
		}
		else {
			pathSegment.color = glm::vec3(0);
			pathSegment.remainingBounces = 0; // no intersection = stop tracing the path
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment &iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

__global__ void copyIntersectionMaterials(int nPaths, int* materials, ShadeableIntersection* intersections) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= nPaths) {
		return;
	}

	materials[index] = intersections[index].materialId;
}

struct path_should_continue
{
	__host__ __device__
		bool operator()(const PathSegment segment)
	{
		return segment.remainingBounces > 0;
	}
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
// Each path-trace call runs 1 iteration
// Each iteration, we add the new results of one ray per pixel to the dev_image
// and send the new dev_image back to opengl
// Why hasn't the image gone to white yet??
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

#if CACHE_FIRST_BOUNCE
	if (iter == 1) { // on first iteration, need to generate new camera rays
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
		checkCUDAError("generate camera ray");
		cudaMemcpy(dev_first_bounce_paths, dev_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	}
	else { // on next iterations, we can used cached values from dev_first_bounce_paths
		cudaMemcpy(dev_paths, dev_first_bounce_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	}
#else
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
#endif

	int depth = 0;
	PathSegment *dev_path_end = dev_paths + pixelcount;
	int num_paths_to_trace = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection)); // TODO: can change this to num paths

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths_to_trace + blockSize1d - 1) / blockSize1d;
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths_to_trace
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
		checkCUDAError("trace one bounce");

		// sort intersections by material type
		// also sort pathSegments in parallel to intersections
#if SORT_BY_MATERIALS
		copyIntersectionMaterials << <numblocksPathSegmentTracing, blockSize1d >> >
			(num_paths_to_trace, dev_intersection_materials, dev_intersections);
		thrust::sort_by_key(thrust::device, dev_intersection_materials, dev_intersection_materials + num_paths_to_trace, dev_paths);

		copyIntersectionMaterials << <numblocksPathSegmentTracing, blockSize1d >> >
			(num_paths_to_trace, dev_intersection_materials, dev_intersections);
		thrust::sort_by_key(thrust::device, dev_intersection_materials, dev_intersection_materials + num_paths_to_trace, dev_intersections);
#endif

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

		shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths_to_trace,
			depth,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_imageSources,
			dev_imageBuffers
			);

		 //update num_paths using stream compaction
		dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_path_end, path_should_continue());
		num_paths_to_trace = dev_path_end - dev_paths;

		if (depth > traceDepth || num_paths_to_trace <= 0) {
			iterationComplete = true;
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
