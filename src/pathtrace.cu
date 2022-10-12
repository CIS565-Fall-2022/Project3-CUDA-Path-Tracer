#include <cstdio>
#include <cuda.h>

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <thrust/remove.h>

#include <thrust/random.h>
#include <thrust/transform.h>

#include <thrust/sort.h>
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.cuh"
#include "interactions.h"
#include "rendersave.h"
#include "Collision/AABB.h"
#include "Octree/octree.h"
#include "consts.h"
#include "Denoise/denoise.cuh"

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

// functors

struct RadianceToNormalizedRGB {
	int iter;
	RadianceToNormalizedRGB(int iter) : iter(iter) { }
	__host__ __device__ color_t operator()(color_t const& r) const {
		if (!iter) {
			return color_t(0);
		}
		return glm::clamp(r / (float) iter, 0.f, 1.f);
	}
};
struct RadianceToRGBA {
	int iter;
	RadianceToRGBA(int iter) : iter(iter) { }
	__host__ __device__ uchar4 operator()(color_t const& r) const {
		if (!iter) {
			return make_uchar4(0, 0, 0, 0);
		}
		return make_uchar4(
			glm::clamp((int)(r.x / iter * 255.f), 0, 255),
			glm::clamp((int)(r.y / iter * 255.f), 0, 255),
			glm::clamp((int)(r.z / iter * 255.f), 0, 255),
			0);
	}
};
struct NormalizedRGBToRGBA {
	__host__ __device__ uchar4 operator()(color_t const& r) const {
		return make_uchar4(
			glm::clamp((int)(r.x * 255.f), 0, 255),
			glm::clamp((int)(r.y * 255.f), 0, 255),
			glm::clamp((int)(r.z * 255.f), 0, 255),
			0);
	}
};
struct NormalToRGBA {
	__host__ __device__ uchar4 operator()(glm::vec3 const& n) const {
		// convert from the range [-1, 1] to [0, 1]
		return make_uchar4(
			glm::clamp((int)(((n.x + 1.f) / 2.f) * 255.f), 0, 255),
			glm::clamp((int)(((n.y + 1.f) / 2.f) * 255.f), 0, 255),
			glm::clamp((int)(((n.z + 1.f) / 2.f) * 255.f), 0, 255),
			0);
	}
};
__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

void PathTracer::unitTest() {  }

static RenderState* renderState = nullptr;
static Scene* hst_scene = nullptr;
static std::string cur_scene = "";

static Span<glm::vec3>             dev_image;
static Span<Geom>                  dev_geoms;
static Span<PathSegment>           dev_paths;
static Span<ShadeableIntersection> dev_intersections;

// static variables for device memory, any extra info you need, etc
// ...
static Span<ShadeableIntersection> dev_cached_intersections;
static Span<Light> dev_lights;
static MeshInfo dev_mesh_info;

static std::vector<TextureGPU> dev_texs;

static octree* tree;
static octreeGPU dev_tree;

GLuint s_pbo_id = -1;
uchar4* s_pbo_dptr = nullptr;

struct DenoiseBuffers {
	DenoiseBuffers() = default;
	DenoiseBuffers(DenoiseBuffers const&) = delete;
	DenoiseBuffers(DenoiseBuffers&&) = delete;

	void init(int pixelcount) {
		rt = make_span<color_t>(pixelcount);
		d = make_span<glm::vec3>(pixelcount);
		n = make_span<glm::vec3>(pixelcount);
		x = make_span<glm::vec3>(pixelcount);
	}
	void free() {
		FREE(rt);
		FREE(n);
		FREE(x);
		FREE(d);
	}

	Span<color_t> rt;
	Span<glm::vec3> n, x, d;
};

// pathtracer state
static bool enable_denoise = false;
static bool render_paused = false;
static int cur_iter;

static DenoiseBuffers denoise_buffers;
static Denoiser::ParamDesc denoise_params;

// helper for displaying a texture for debugging purposes
struct DebugTexScope {
	DebugTexScope(DebugTexScope const&) = delete;
	DebugTexScope(DebugTexScope&&) = delete;

	DebugTexScope(DebugTextureType type, uchar4* pbo) {
		glm::vec3 const* tex;
		switch (type) {
		case DebugTextureType::DIFFUSE_BUF: tex = denoise_buffers.d; break;
		case DebugTextureType::NORM_BUF: tex = denoise_buffers.n; break;
		case DebugTextureType::POS_BUF: tex = denoise_buffers.x; break;
		default: throw;
		}
		render_paused = true;
		Camera const& cam = hst_scene->state.camera;
		int pixelcount = cam.resolution.x * cam.resolution.y;

		if (type == DebugTextureType::NORM_BUF) {
			thrust::transform(thrust::device, tex, tex + pixelcount, pbo, NormalToRGBA());
		} else {
			thrust::transform(thrust::device, tex, tex + pixelcount, pbo, NormalizedRGBToRGBA());
		}
	}

	~DebugTexScope() {
		render_paused = false;
	}
};
static DebugTexScope* s_debug_tex_scope = nullptr;

void PathTracer::beginFrame(unsigned int pbo_id) {
	s_pbo_id = pbo_id;
	CHECK_CUDA(cudaGLMapBufferObject((void**)&s_pbo_dptr, s_pbo_id));
}

void PathTracer::endFrame() {
	CHECK_CUDA(cudaGLUnmapBufferObject(s_pbo_id));
}

void PathTracer::pathtraceInit(Scene* scene, RenderState* state, bool force_change) {
	if (!scene) throw;
	bool scene_changed = force_change || cur_scene != scene->filename;
	hst_scene = scene;
	cur_scene = scene->filename;
	renderState = state;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	dev_image = make_span(state->image);
	dev_paths = make_span<PathSegment>(pixelcount);
	dev_intersections = make_span<ShadeableIntersection>(pixelcount);
#ifdef CACHE_FIRST_BOUNCE
	dev_cached_intersections = make_span<ShadeableIntersection>(pixelcount);
#endif // CACHE_FIRST_BOUNCE

	if (scene_changed) {
		denoise_buffers.init(pixelcount);

		dev_geoms = make_span(scene->geoms);
		dev_lights = make_span(scene->lights);
		dev_mesh_info.vertices = make_span(scene->vertices);
		dev_mesh_info.normals = make_span(scene->normals);
		dev_mesh_info.uvs = make_span(scene->uvs);
		dev_mesh_info.tris = make_span(scene->triangles);
		dev_mesh_info.meshes = make_span(scene->meshes);
		dev_mesh_info.tangents = make_span(scene->tangents);
		dev_mesh_info.materials = make_span(scene->materials);

		for (Texture const& hst_tex : scene->textures) {
			TextureGPU dev_tex(hst_tex);
			dev_texs.push_back(dev_tex);
		}
		dev_mesh_info.texs = make_span(dev_texs);
#ifdef OCTREE_CULLING
		tree = new octree(*scene, scene->world_AABB, OCTREE_DEPTH);
		dev_tree.init(*tree, dev_mesh_info, dev_geoms);
#endif // OCTREE_CULLING
	}
    checkCUDAError("pathtraceInit");
}

void PathTracer::pathtraceFree(Scene* scene, bool force_change) {
	bool scene_changed = force_change || !scene || cur_scene != scene->filename;

	if (s_debug_tex_scope) {
		delete s_debug_tex_scope;
		s_debug_tex_scope = nullptr;
	}

	FREE(dev_image);
	FREE(dev_paths);
	FREE(dev_intersections);
#ifdef CACHE_FIRST_BOUNCE
	FREE(dev_cached_intersections);
#endif // CACHE_FIRST_BOUNCE

	if (scene_changed) {
		denoise_buffers.free();

		FREE(dev_geoms);
		FREE(dev_lights);
		FREE(dev_mesh_info.vertices);
		FREE(dev_mesh_info.normals);
		FREE(dev_mesh_info.uvs);
		FREE(dev_mesh_info.tris);
		FREE(dev_mesh_info.meshes);
		FREE(dev_mesh_info.tangents);
		FREE(dev_mesh_info.materials);
		for (TextureGPU& tex : dev_texs) {
			tex.free();
		}
		dev_texs.clear();
		FREE(dev_mesh_info.texs);

#ifdef OCTREE_CULLING
		dev_tree.free();
		delete tree;
#endif
	}
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

// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int offset,
	Span<PathSegment> paths,
	Span<Geom> geoms,
	ShadeableIntersection* intersections,
	MeshInfo meshInfo,
	ShadeableIntersection* cache_intersections,
	octreeGPU octree)
{
	int path_index = offset + blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index >= paths.size()) {
		return;
	}
	PathSegment path = paths[path_index];

#ifndef COMPACTION
	if (!path.remainingBounces) {
		return;
	}
#endif // COMPACTION
	assert(path.remainingBounces > 0);
	ShadeableIntersection& inters = intersections[path_index];

#ifdef OCTREE_CULLING
	if (!octree.search(inters, path.ray)) {
		inters.t = -1;
	}
#else
	float t_min = FLT_MAX;
	inters.t = -1;


	for (int i = 0; i < geoms.size(); i++) {
		Geom& geom = geoms[i];

#ifdef AABB_CULLING
		if (!AABBRayIntersect(geom.bounds, path.ray, nullptr))
			continue;
#endif // AABB_CULLING

		float t;
		ShadeableIntersection tmp;

		if (geom.type == CUBE) {
			t = boxIntersectionTest(geom, path.ray, tmp);
		} else if (geom.type == SPHERE) {
			t = sphereIntersectionTest(geom, path.ray, tmp);
		} else if (geom.type == MESH) {
			t = meshIntersectionTest(geom, path.ray, meshInfo, tmp);
		}
		// add more intersection tests here... triangle? metaball? CSG?

		// Compute the minimum t from the intersection tests to determine what
		// scene geometry object was hit first.
		if (t > 0.0f && t_min > t) {
			t_min = t;
			inters = tmp;
		}
	}
#endif // OCTREE_CULLING

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
int PathTracer::pathtrace(int iter) {
	cur_iter = iter;
	if (render_paused) {
		return iter;
	}

	++cur_iter;

    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
	dim3 blk_per_grid2d(DIV_UP(cam.resolution.x, 8), DIV_UP(cam.resolution.y, 8));
	dim3 blk_sz2d(8,8);

    generateRayFromCamera KERN_PARAM(blk_per_grid2d, blk_sz2d) (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

	for (int depth = 0, num_paths = pixelcount; num_paths > 0 && depth < traceDepth; ++depth) {
		// clean shading chunks
		MEMSET(dev_intersections, 0, num_paths);

		ShadeableIntersection* dev_cached_inters;
		ShadeableIntersection* dev_inters;

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

		// split dev_paths [0 : num_paths] into chunks
#ifndef MAX_INTERSECTION_TEST_SIZE // if not defined, launch all paths at once
#define MAX_INTERSECTION_TEST_SIZE num_paths
#endif // !MAX_INTERSECTION_TEST_SIZE

		for (int i = 0; i < num_paths; i += MAX_INTERSECTION_TEST_SIZE) {
			int j = std::min(num_paths, i + MAX_INTERSECTION_TEST_SIZE);
			int size = j - i;

			computeIntersections KERN_PARAM(DIV_UP(size, BLOCK_SIZE), BLOCK_SIZE) (
				i,
				dev_paths.subspan(0, num_paths),
				dev_geoms,
				dev_inters,
				dev_mesh_info,
				dev_cached_inters,
				dev_tree
			);
			
			checkCUDAError(std::string("trace one bounce, inters size = " +
				std::to_string(MAX_INTERSECTION_TEST_SIZE)).c_str());
			cudaDeviceSynchronize();
		}

		// initialize position and normal buffers for denoising
		// NOTE: must do this before the material sorting
		if (!depth && !iter) {
			// normal
			thrust::transform(
				thrust::device,
				dev_inters,
				dev_inters + pixelcount,
				denoise_buffers.n.get(),
				Denoiser::IntersectionToNormal());

			// position
			thrust::transform(
				thrust::device,
				dev_inters,
				dev_inters + pixelcount,
				denoise_buffers.x.get(),
				Denoiser::IntersectionToPos());

			// diffuse
			thrust::transform(
				thrust::device,
				dev_inters,
				dev_inters + pixelcount,
				denoise_buffers.d.get(),
				Denoiser::IntersectionToDiffuse(dev_mesh_info.materials));
		}

		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by evaluating the BSDF.
#ifdef SORT_MAT
		thrust::sort_by_key(thrust::device, dev_inters, dev_inters + num_paths, dev_paths.get());
#endif

#ifdef FAKE_SHADE
#define shadeMaterial shadeFakeMaterial
#endif
		shadeMaterial KERN_PARAM(DIV_UP(num_paths, BLOCK_SIZE), BLOCK_SIZE) (
			iter,
			dev_paths.subspan(0, num_paths),
			dev_lights,
			dev_inters,
			dev_mesh_info.materials
		);

		checkCUDAError("shadeMaterial");
		cudaDeviceSynchronize();

#ifdef COMPACTION
		num_paths = thrust::partition(thrust::device, dev_paths.get(), dev_paths.get() + num_paths, PathSegment::PartitionRule()) - dev_paths.get();
#endif // COMPACTION

#ifdef MAX_DEPTH_OVERRIDE
		if (depth == MAX_DEPTH_OVERRIDE)
			break;
#endif
	}
	
	// Assemble this iteration and apply it to the image
    finalGather KERN_PARAM(DIV_UP(pixelcount, BLOCK_SIZE), BLOCK_SIZE) (pixelcount, dev_image, dev_paths);
	cudaDeviceSynchronize();

	// denoise
	if (enable_denoise) {
		thrust::transform(
			thrust::device, 
			dev_image.get(),
			dev_image.get() + pixelcount,
			denoise_buffers.rt.get(),
			RadianceToNormalizedRGB(cur_iter));

		Denoiser::denoise(denoise_buffers.rt, denoise_buffers.n, denoise_buffers.x, denoise_buffers.d, denoise_params);

		thrust::transform(
			thrust::device,
			denoise_buffers.rt.get(),
			denoise_buffers.rt.get() + pixelcount,
			s_pbo_dptr,
			NormalizedRGBToRGBA()
		);
	} else {
		thrust::transform(
			thrust::device,
			dev_image.get(),
			dev_image.get() + pixelcount,
			s_pbo_dptr,
			RadianceToRGBA(cur_iter));
	}

    ///////////////////////////////////////////////////////////////////////////
    // Retrieve image from GPU
	D2H(hst_scene->state.image.data(), dev_image, pixelcount);

    checkCUDAError("pathtrace");

	return cur_iter;
}

bool PathTracer::saveRenderState(char const* filename) {
	return save_state(cur_iter, *renderState, *hst_scene, filename);
}

void PathTracer::togglePause() {
	if (s_debug_tex_scope) {
		// if a debug texture is being displayed, do nothing
		return;
	}
	render_paused = !render_paused;
}

bool PathTracer::isPaused() {
	return render_paused;
}

void PathTracer::enableDenoise() {
	enable_denoise = true;
}
void PathTracer::disableDenoise() {
	enable_denoise = false;
}
octreeGPU PathTracer::getTree() {
	return dev_tree;
}
void PathTracer::setDenoise(Denoiser::ParamDesc const& desc) {
	denoise_params = desc;
}

void PathTracer::debugTexture(DebugTextureType type) {
	if (s_debug_tex_scope) {
		delete s_debug_tex_scope;
	}
	switch (type) {
	case DebugTextureType::NORM_BUF:
	case DebugTextureType::POS_BUF:
	case DebugTextureType::DIFFUSE_BUF:
		s_debug_tex_scope = new DebugTexScope(type, s_pbo_dptr);
		return;
	case DebugTextureType::NONE:
	default:
		render_paused = false;
		s_debug_tex_scope = nullptr;
		return;
	}
}