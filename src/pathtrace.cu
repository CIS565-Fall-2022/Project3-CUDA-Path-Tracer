#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "rendersave.h"
#include "Collision/AABB.h"
#include "Octree/octree.h"

// impl switches
#define COMPACTION
#define SORT_MAT
#define AABB_CULLING

#define OCTREE_CULLING
// #define DEPTH_OF_FIELD

// #define ANTI_ALIAS_JITTER
// #define FAKE_SHADE

#define CACHE_FIRST_BOUNCE
#if (defined(CACHE_FIRST_BOUNCE) && defined(ANTI_ALIAS_JITTER)) || (defined(CACHE_FIRST_BOUNCE) && defined(DEPTH_OF_FIELD)) 
#error "ANTI_ALIAS_JITTER or CACHE_FIRST_BOUNCE cannot be used with CACHE_FIRST_BOUNCE"
#endif


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

__device__ int f(int i) {
	if (!i) return 1;
	return f(i - 1) * i;
}
__global__ void recur_test() {
	printf("recur_test: %d", f(5));
}

__global__ void print2d(Span<Span<int>> arr) {
	for (int i = 0; i < arr.size(); ++i) {
		for (int j = 0; j < arr[i].size(); ++j) {
			printf("%d ", arr[i][j]);
		}
		printf("\n");
	}
}
void PathTracer::unitTest() {
#ifndef NDEBUG
	// test 2d span
#define SECTION(name) std::cout << "========= " << name << " =========\n";
	SECTION("test 2d array") {
		thrust::default_random_engine rng = makeSeededRandomEngine(0, 0, 0);
		thrust::uniform_int_distribution<int> udist(3, 10);
		std::vector<std::vector<int>> jagged(udist(rng));
		for (int i = 0; i < jagged.size(); ++i) {
			jagged[i].resize(udist(rng));
			for (int j = 0; j < jagged[i].size(); ++j) {
				jagged[i][j] = (i + 1) * (j + 1);
			}
		}

		std::vector<Span<int>> dev_arrs;
		for (auto const& v : jagged) {
			int* arr;
			ALLOC(arr, v.size());
			H2D(arr, v.data(), v.size());
			dev_arrs.emplace_back(v.size(), arr);
		}

		Span<int>* arrs;
		ALLOC(arrs, jagged.size());
		H2D(arrs, dev_arrs.data(), jagged.size());

		print2d KERN_PARAM(1, 1) ( { (int)jagged.size(), arrs } );
		for (int i = 0; i < jagged.size(); ++i) {
			FREE(dev_arrs[i]);
		}
		FREE(arrs);
	}
	
	SECTION("recursion test") {
		recur_test KERN_PARAM(1, 1) ();
	}
#endif // !NDEBUG
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

static RenderState* renderState = nullptr;
static Scene* hst_scene = nullptr;

static Span<glm::vec3>             dev_image;
static Span<Geom>                  dev_geoms;
static Span<Material>              dev_materials;
static Span<PathSegment>           dev_paths;
static Span<ShadeableIntersection> dev_intersections;

// static variables for device memory, any extra info you need, etc
// ...
static Span<ShadeableIntersection> dev_cached_intersections;
static Span<Light> dev_lights;
static MeshInfo dev_mesh_info;

static thrust::device_ptr<PathSegment> dev_thrust_paths;
static std::vector<TextureGPU> dev_texs;

// TODO
static octree* tree;
static octreeGPU dev_tree;

// pathtracer state
static bool render_paused = false;
static int cur_iter;

void PathTracer::pathtraceInit(Scene* scene, RenderState* state) {
	hst_scene = scene;
	renderState = state;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	dev_image = make_span(state->image);
	dev_paths = make_span<PathSegment>(pixelcount);
	dev_materials = make_span(scene->materials);
	dev_intersections = make_span<ShadeableIntersection>(pixelcount);
	dev_geoms = make_span(scene->geoms);
	dev_lights = make_span(scene->lights);
	dev_mesh_info.vertices = make_span(scene->vertices);
	dev_mesh_info.normals = make_span(scene->normals);
	dev_mesh_info.uvs = make_span(scene->uvs);
	dev_mesh_info.tris = make_span(scene->triangles);
	dev_mesh_info.meshes = make_span(scene->meshes);
	dev_mesh_info.tangents = make_span(scene->tangents);
	for (Texture const& hst_tex : scene->textures) {
		TextureGPU dev_tex(hst_tex);
		dev_texs.push_back(dev_tex);
	}
	dev_mesh_info.texs = make_span(dev_texs);
	dev_thrust_paths = thrust::device_ptr<PathSegment>((PathSegment*)dev_paths);


#ifdef CACHE_FIRST_BOUNCE
	dev_cached_intersections = make_span<ShadeableIntersection>(pixelcount);
#endif // CACHE_FIRST_BOUNCE

    checkCUDAError("pathtraceInit");
}

void PathTracer::pathtraceFree() {
	FREE(dev_image);
	FREE(dev_paths);
	FREE(dev_geoms);
	FREE(dev_materials);
	FREE(dev_intersections);
	FREE(dev_lights);
	FREE(dev_mesh_info.vertices);
	FREE(dev_mesh_info.normals);
	FREE(dev_mesh_info.uvs);
	FREE(dev_mesh_info.tris);
	FREE(dev_mesh_info.meshes);
	FREE(dev_mesh_info.tangents);
	for (TextureGPU& tex : dev_texs) {
		tex.free();
	}
	dev_texs.clear();
	FREE(dev_mesh_info.texs);

#ifdef CACHE_FIRST_BOUNCE
	FREE(dev_cached_intersections);
#endif // CACHE_FIRST_BOUNCE

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

#ifdef ANTI_ALIAS_JITTER
		// randomly jitter the ray
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
		thrust::uniform_real_distribution<float> udist(-0.5f, 0.5f);
		x += udist(rng);
		y += udist(rng);
#endif // ANTI_ALIAS

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
	Span<PathSegment> paths,
	Span<Geom> geoms,
	ShadeableIntersection* intersections,
	Material* materials,
	MeshInfo meshInfo,
	ShadeableIntersection* cache_intersections)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index >= paths.size()) {
		return;
	}
	PathSegment pathSegment = paths[path_index];

#ifndef COMPACTION
	if (!pathSegment.remainingBounces) {
		return;
	}
#endif // COMPACTION

	assert(pathSegment.remainingBounces > 0);

	float t_min = FLT_MAX;

	// naive parse through global geoms
	ShadeableIntersection& inters = intersections[path_index];
	inters.t = -1;

	for (int i = 0; i < geoms.size(); i++) {
		Geom& geom = geoms[i];
#ifdef AABB_CULLING
		if (!intersect(geom.bounds, pathSegment.ray))
			continue;
#endif // AABB_CULLING

		float t;
		ShadeableIntersection tmp;

		if (geom.type == CUBE) {
			t = boxIntersectionTest(geom, pathSegment.ray, tmp);
		} else if (geom.type == SPHERE) {
			t = sphereIntersectionTest(geom, pathSegment.ray, tmp);
		} else if (geom.type == MESH) {
			t = meshIntersectionTest(geom, pathSegment.ray, materials, meshInfo, tmp);
		}
		// add more intersection tests here... triangle? metaball? CSG?

		// Compute the minimum t from the intersection tests to determine what
		// scene geometry object was hit first.
		if (t > 0.0f && t_min > t) {
			t_min = t;
			inters = tmp;
			inters.t = t;
		}
	}

	if (cache_intersections) {
		cache_intersections[path_index] = inters;
	}
}

__global__ void shadeMaterial(
	int iter,
	Span<PathSegment> paths,
	Span<Light> lights,
	ShadeableIntersection* shadeableIntersections,
	Material* materials) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= paths.size()) {
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
		glm::vec3 materialColor = material.diffuse;

		// If the material indicates that the object was a light, "light" the ray
		if (material.emittance > 0.0f) {
			path.color *= (materialColor * material.emittance);
			path.terminate();
		} else {
			scatterRay(path, intersection, material, lights, rng);
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
	int iter,
	Span<PathSegment> paths,
	Span<Light> lights,
	ShadeableIntersection* shadeableIntersections,
	Material* materials) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < paths.size())
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
			glm::vec3 materialColor;
			if (material.textures.diffuse != -1) {
				materialColor = intersection.tex_color;
			} else {
				materialColor = material.diffuse;
			}

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
__global__ void finalGather(int numPixels, glm::vec3* image, PathSegment* iterationPaths) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < numPixels) {
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 * 
 * Returns the new iteration
 */
int PathTracer::pathtrace(uchar4 *pbo, int iter) {
	cur_iter = iter;
	if (render_paused) {
		return iter;
	}

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

		ShadeableIntersection* dev_cached_inters;
		ShadeableIntersection* dev_inters;
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

		// tracing
#ifdef CACHE_FIRST_BOUNCE
		if (!depth && !iter) {
			// fill the cache
			dev_inters = dev_intersections;
			dev_cached_inters = dev_cached_intersections;
		} else if (!depth) {
			// use cached bounces for the first depth
			dev_inters = dev_cached_intersections;
			dev_cached_inters = nullptr;
		} else {
			// intersect as usual
			dev_inters = dev_intersections;
			dev_cached_inters = nullptr;
		}
#else
		dev_inters = dev_intersections;
		dev_cached_inters = nullptr;
#endif
		computeIntersections KERN_PARAM(numblocksPathSegmentTracing, blockSize1d) (
			depth,
			dev_paths.subspan(0, num_paths),
			dev_geoms,
			dev_inters,
			dev_materials,
			dev_mesh_info,
			dev_cached_inters
		);

		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();

		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
#ifdef SORT_MAT
		thrust::device_ptr<ShadeableIntersection> dev_thrust_inters(dev_inters);
		thrust::sort_by_key(dev_thrust_inters, dev_thrust_inters + num_paths, dev_thrust_paths);
#endif

#ifdef FAKE_SHADE
#define shadeMaterial shadeFakeMaterial
#endif
		shadeMaterial KERN_PARAM(numblocksPathSegmentTracing, blockSize1d) (
			iter,
			dev_paths.subspan(0, num_paths),
			dev_lights,
			dev_inters,
			dev_materials
		);

		checkCUDAError("shadeMaterial");
		cudaDeviceSynchronize();

#ifdef COMPACTION
		num_paths = thrust::partition(dev_thrust_paths, dev_thrust_paths + num_paths, PathSegment::PartitionRule()) - dev_thrust_paths;
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

	return cur_iter = iter + 1;
}

bool PathTracer::saveRenderState(char const* filename) {
	return save_state(cur_iter, *renderState, *hst_scene, filename);
}

void PathTracer::togglePause() {
	render_paused = !render_paused;
}

bool PathTracer::isPaused() {
	return render_paused;
}