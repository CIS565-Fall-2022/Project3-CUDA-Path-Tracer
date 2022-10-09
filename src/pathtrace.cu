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

#define TIMING 0

#define ERRORCHECK 1
#define SORT_BY_MATERIAL 0
#define FIRST_BOUNCE_CACHE 1
#define ANTIALIASING 0
#define DIRECT_LIGHTING 1

#define DEPTH_OF_FIELD 0

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

#if TIMING
// Timing
static cudaEvent_t event_start = nullptr;
static cudaEvent_t event_end = nullptr;
#endif

static int* dev_stencil = NULL;
static ShadeableIntersection* dev_first_intersections = NULL;

static Texture* dev_textures;
static glm::vec3* dev_colorTexture;
static glm::vec3* dev_normalTexture;
static glm::vec3* dev_emissiveTexture;

static glm::vec4* dev_lightSourceSampled;

static Triangle* dev_triangles;
static SceneMeshesData dev_sceneMeshesData;

// Octree
static int* dev_triangleIndice;
static OctreeNode* dev_octree;

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
	cudaMalloc(&dev_stencil, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_stencil, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_first_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_first_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_sceneMeshesData.indices, scene->indices.size() * sizeof(unsigned short));
	cudaMemcpy(dev_sceneMeshesData.indices, scene->indices.data(), scene->indices.size() * sizeof(unsigned short), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_sceneMeshesData.positions, scene->positions.size() * sizeof(glm::vec3));
	cudaMemcpy(dev_sceneMeshesData.positions, scene->positions.data(), scene->positions.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_sceneMeshesData.normals, scene->normals.size() * sizeof(glm::vec3));
	cudaMemcpy(dev_sceneMeshesData.normals, scene->normals.data(), scene->normals.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_sceneMeshesData.normals, scene->normals.size() * sizeof(glm::vec3));
	cudaMemcpy(dev_sceneMeshesData.normals, scene->normals.data(), scene->normals.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_sceneMeshesData.uvs, scene->uvs.size() * sizeof(glm::vec2));
	cudaMemcpy(dev_sceneMeshesData.uvs, scene->uvs.data(), scene->uvs.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_sceneMeshesData.tangents, scene->tangents.size() * sizeof(glm::vec4));
	cudaMemcpy(dev_sceneMeshesData.tangents, scene->tangents.data(), scene->tangents.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
	cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_colorTexture, scene->colorTexture.size() * sizeof(glm::vec3));
	cudaMemcpy(dev_colorTexture, scene->colorTexture.data(), scene->colorTexture.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_normalTexture, scene->normalTexture.size() * sizeof(glm::vec3));
	cudaMemcpy(dev_normalTexture, scene->normalTexture.data(), scene->normalTexture.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_emissiveTexture, scene->emissiveTexture.size() * sizeof(glm::vec3));
	cudaMemcpy(dev_emissiveTexture, scene->emissiveTexture.data(), scene->emissiveTexture.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_lightSourceSampled, pixelcount * sizeof(glm::vec4));
	cudaMemset(dev_lightSourceSampled, 0, pixelcount * sizeof(glm::vec4));

	// Octree
	cudaMalloc(&dev_triangleIndice, scene->trianglesIndices.size() * sizeof(int));
	cudaMemcpy(dev_triangleIndice, scene->trianglesIndices.data(), scene->trianglesIndices.size() * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_octree, scene->octree.size() * sizeof(OctreeNode));
	cudaMemcpy(dev_octree, scene->octree.data(), scene->octree.size() * sizeof(OctreeNode), cudaMemcpyHostToDevice);

#if TIMING
	cudaEventCreate(&event_start);
	cudaEventCreate(&event_end);
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
	cudaFree(dev_stencil);
	cudaFree(dev_first_intersections);

	cudaFree(dev_triangles);

	cudaFree(dev_sceneMeshesData.indices);
	cudaFree(dev_sceneMeshesData.positions);
	cudaFree(dev_sceneMeshesData.normals);
	cudaFree(dev_sceneMeshesData.uvs);
	cudaFree(dev_sceneMeshesData.tangents);

	cudaFree(dev_textures);
	cudaFree(dev_colorTexture);
	cudaFree(dev_normalTexture);
	cudaFree(dev_emissiveTexture);

	cudaFree(dev_lightSourceSampled);

	cudaFree(dev_octree);
	cudaFree(dev_triangleIndice);

#if TIMING
	if (event_start != NULL)
		cudaEventDestroy(event_start);
	if (event_end != NULL)
		cudaEventDestroy(event_end);
#endif

	checkCUDAError("pathtraceFree");
}

__device__
glm::vec2 ConcentricSampleDisk(thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);

	glm::vec2 uOffset;
	uOffset.x = 2.f * u01(rng) - 1;
	uOffset.y = 2.f * u01(rng) - 1;

	// << Handle degeneracy at the origin >>
	if (uOffset.x == 0 && uOffset.y == 0)
		return glm::vec2(0.f, 0.f);

	// << Apply concentric mapping to point >>
	float theta, r;
	if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
		r = uOffset.x;
		theta = (PI / 4.f) * (uOffset.y / uOffset.x);
	}
	else {
		r = uOffset.y;
		theta = (PI / 2.f) - (PI / 4.f) * (uOffset.x / uOffset.y);
	}
	return r * glm::vec2(std::cos(theta), std::sin(theta));
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
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, iter + 7, iter * 7);

		// TODO: implement antialiasing by jittering the ray
#if ANTIALIASING && !(FIRST_BOUNCE_CACHE)
		thrust::uniform_real_distribution<float> u01(-0.5, 0.5);

		float sampleX = u01(rng);
		float sampleY = u01(rng);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - sampleX - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - sampleY - (float)cam.resolution.y * 0.5f)
		);
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif

#if DEPTH_OF_FIELD && !(FIRST_BOUNCE_CACHE)
		if (cam.aperture > 0) {
			// <<Sample point on lens>> 
			glm::vec2 pLens = cam.aperture * ConcentricSampleDisk(rng);

			//<< Compute point on plane of focus >>
			float ft = abs(cam.focal / segment.ray.direction.z);
			glm::vec3 pFocus = getPointOnRay(segment.ray, ft);

			// << Update ray for effect of lens >>
			segment.ray.origin += glm::vec3(pLens.x, pLens.y, 0);
			segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
		}
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
	, int geoms_size
	, SceneMeshesData sceneMeshesData
	, Triangle* triangles
	, ShadeableIntersection* intersections
	, glm::vec3* normalMap
	, Texture* textureData
	, int normalTextureID
	, int* triangleIndice
	, OctreeNode* octree
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
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv;
		glm::vec4 tmp_tangent;
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
			else if (geom.type == MESH) {
				t = meshIntersectionTest(geom, triangleIndice, octree, sceneMeshesData, triangles, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, tmp_tangent);
			}

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				uv = tmp_uv;
				normal = tmp_normal;

				if (geom.normalMapID != -1 && geom.useTex && geom.hasTangent) {
					glm::vec3 normalMapValue;
					FetchTexture(textureData[geom.normalMapID], textureData[geom.normalMapID].offsetNormal, normalMap, uv, normalMapValue);
					normalMapValue = (2.f * normalMapValue - 1.f);
					//normal += normalMapValue;
					glm::vec3 outputNormal;
					normalMapping(normal, normalMapValue, tmp_tangent, outputNormal);
					normal = outputNormal;
				}
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
		}
	}
}

__global__ void directlyLighting(
	int iter,
	int num_paths,
	int num_lightSource, 
	int depth,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	glm::vec4* lightSourceSampled
	) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		if (pathSegments[idx].remainingBounces != 1) {
			return;
		}

		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f && pathSegments[idx].isDifuse) { // if the intersection exists...
			// Set up the RNG
			// LOOK: this is how you use thrust's RNG! Please look at
			// makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
			thrust::uniform_real_distribution<float> u01(0, num_lightSource);
			int lightSourceIdx = int(u01(rng));
		
			glm::vec3 intersectionPoint = getPointOnRay(pathSegments[idx].ray, intersection.t);
			pathSegments[idx].ray.direction = glm::normalize(glm::vec3(lightSourceSampled[lightSourceIdx]) - pathSegments[idx].ray.origin);
			pathSegments[idx].remainingBounces = 1;
			//pathSegments[idx].color = glm::vec3(1.f, 0.f, 1.f);
		}
	}

}

__global__ void shadeMaterialBSDF(
	int iter,
	int num_paths,
	int num_lightSource,
	int depth,
	int traceDepth,
	int* stencil,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials,
	glm::vec4* lightSourceSampled,
	Texture* texData, glm::vec3* colorTexture, glm::vec3* emissiveTexture
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		if (pathSegments[idx].remainingBounces <= 0) {
			return;
		}

		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			// Set up the RNG
			// LOOK: this is how you use thrust's RNG! Please look at
			// makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
				lightSourceSampled[idx] = glm::vec4(getPointOnRay(pathSegments[idx].ray, intersection.t), 1);
				pathSegments[idx].remainingBounces = 0;
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else{
				stencil[idx] = 1;
				glm::vec3 intersectionPoint = getPointOnRay(pathSegments[idx].ray, intersection.t);
				scatterRay(idx, pathSegments[idx], intersectionPoint, intersection.surfaceNormal, intersection.uv, material, 
					num_lightSource, lightSourceSampled,
					texData, colorTexture, emissiveTexture, 
					traceDepth, rng);

				pathSegments[idx].remainingBounces--;

				//if (pathSegments[idx].remainingBounces == 0) {
				//	pathSegments[idx].color = glm::vec3(1.f, 0.f, 1.f);
				//}
			}

			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
			return;
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

struct is_dead {
	__host__ __device__
		bool operator()(const PathSegment& path) {
		return path.remainingBounces == 0;
	}
};

struct is_alive {
	__host__ __device__
		bool operator()(const int x) {
		return x > 0;
	}
};

struct isEmpty {
	__host__ __device__
		bool operator()(const glm::vec4& p) {
		return p.w <= 0;
	}
};

struct MatCmp {
	__host__ __device__
		bool operator()(const ShadeableIntersection& inter1, const ShadeableIntersection& inter2) {
		return inter1.materialId > inter2.materialId;
	}
};

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
	
	glm::vec4* dev_lightSourceSampled_end = dev_lightSourceSampled + pixelcount;
	int num_lightSourceSampled = dev_lightSourceSampled_end - dev_lightSourceSampled;

	//thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
#if TIMING
	cudaEventRecord(event_start);
#endif
	bool iterationComplete = false;
	while (!iterationComplete) {

		ShadeableIntersection* intersections = dev_intersections;

		// clean shading chunks
		cudaMemset(intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		cudaMemset(dev_stencil, 0, pixelcount * sizeof(ShadeableIntersection));
		//cudaMemset(dev_lightSourceSampled, 0, pixelcount * sizeof(glm::vec3));

		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if FIRST_BOUNCE_CACHE
		if (iter != 1 && depth == 0) {

			intersections = dev_first_intersections;
		}
		else {
			// tracing
			
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_sceneMeshesData
				, dev_triangles
				, intersections
				, dev_normalTexture
				, dev_textures
				, hst_scene->normalTextureID
				, dev_triangleIndice
				, dev_octree
				);
			checkCUDAError("trace one bounce");

			if (iter == 1 && depth == 0) {
				cudaMemcpy(dev_first_intersections, intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}

#if SORT_BY_MATERIAL
			thrust::sort_by_key(thrust::device, intersections, intersections + num_paths, dev_paths, MatCmp());
#endif // SORT_BY_MATERIAL
		}
#else
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_sceneMeshesData
			, dev_triangles
			, intersections
			, dev_normalTexture
			, dev_textures
			, hst_scene->normalTextureID
			, dev_triangleIndice
			, dev_octree
			);
		checkCUDAError("trace one bounce");

#if SORT_BY_MATERIAL
		thrust::sort_by_key(thrust::device, intersections, intersections + num_paths, dev_paths, MatCmp());
#endif // SORT_BY_MATERIAL

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

		shadeMaterialBSDF << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			num_lightSourceSampled,
			depth,
			traceDepth,
			dev_stencil,
			intersections,
			dev_paths,
			dev_materials,
			dev_lightSourceSampled,
			dev_textures,
			dev_colorTexture,
			dev_emissiveTexture
			);
		checkCUDAError("BSDF ERROR");

		cudaDeviceSynchronize();
#if DIRECT_LIGHTING
		dev_lightSourceSampled_end = thrust::remove_if(thrust::device, dev_lightSourceSampled, dev_lightSourceSampled + num_lightSourceSampled, isEmpty());
		num_lightSourceSampled = dev_lightSourceSampled_end - dev_lightSourceSampled;

		//cout << "Light Source Sampled Number: " << num_lightSourceSampled << " / " << pixelcount << endl;

		directlyLighting << <numblocksPathSegmentTracing, blockSize1d >> > (iter, num_paths, num_lightSourceSampled, depth,
			intersections, dev_paths, dev_lightSourceSampled);
#endif

		dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, dev_stencil, thrust::identity<int>());
		num_paths = dev_path_end - dev_paths;

		iterationComplete = (num_paths <= 0) || (depth > traceDepth); // TODO: should be based off stream compaction results.


		//std::cout << "Iter: " << iter << "  Number of Paths: " << num_paths << std::endl;

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}
#if TIMING
	cudaEventRecord(event_end);
	cudaEventSynchronize(event_end);
	float ms;
	cudaEventElapsedTime(&ms, event_start, event_end);
	std::cout << "Iter: " << iter << " Depth: " << depth << " " << ms << " ms Number of path: " << num_paths << endl;
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
