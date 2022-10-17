#include <cstdio>
#include <cuda.h>

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

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

#include "utilities.h"
#include "pathtrace.h"
#include "intersections.cuh"
#include "interactions.h"
#include "rendersave.h"
#include "Collision/AABB.h"
#include "Octree/octree.h"
#include "consts.h"
#include "Denoise/denoise.cuh"
#include "Profile/pathtracer_profile.h"

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

static std::unique_ptr<octree> tree;
static std::unique_ptr<octreeGPU> dev_tree;

static GLuint s_pbo_id = 0;
static uchar4* s_pbo_dptr = nullptr;

// pathtracer state
static bool enable_denoise = false;
static bool render_paused = false;
static bool texture_debug_active = false;
static int cur_iter;

// copy of the radiance buffer for denoising
color_t* denoise_image;
static Denoiser::DenoiseBuffers denoise_buffers;
static std::unique_ptr<Denoiser::ParamDesc> denoise_params;

// profiling
std::unordered_map<std::string, Profiling::ProfileData> s_prof_data;
std::unordered_map<std::string, Profiling::ProfileData>&
PathTracer::GetProfileData() {
	return s_prof_data;
}

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

	denoise_image = make_span(state->image);

	if (scene_changed) {
		denoise_buffers.init(cam.resolution.x, cam.resolution.y);

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
		tree = std::make_unique<octree>(*scene, scene->world_AABB, OCTREE_DEPTH);
		dev_tree = std::make_unique<octreeGPU>(*tree, dev_mesh_info, dev_geoms);
#endif // OCTREE_CULLING
	}
    checkCUDAError("pathtraceInit");
}

void PathTracer::pathtraceFree(Scene* scene, bool force_change) {
	bool scene_changed = force_change || !scene || cur_scene != scene->filename;

	FREE(dev_image);
	FREE(dev_paths);
	FREE(dev_intersections);
#ifdef CACHE_FIRST_BOUNCE
	FREE(dev_cached_intersections);
#endif // CACHE_FIRST_BOUNCE
	FREE(denoise_image);

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

		pathSegments[index].init(traceDepth, index, cam_ray);
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
	if (path.remainingBounces <= 0) {
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

#ifndef COMPACTION
	if (path.remainingBounces <= 0) {
		return;
	}
#endif // COMPACTION

	assert(path.remainingBounces > 0);


	ShadeableIntersection intersection = shadeableIntersections[idx];
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

#ifndef COMPACTION
		if (path.remainingBounces <= 0) {
			return;
		}
#endif // COMPACTION

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

	ProfileHelper frame_profiling("frame");

    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
	dim3 blk_per_grid2d(DIV_UP(cam.resolution.x, 8), DIV_UP(cam.resolution.y, 8));
	dim3 blk_sz2d(8,8);

	frame_profiling.call(generateRayFromCamera, blk_per_grid2d, blk_sz2d, 
		cam, iter, traceDepth, dev_paths);
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

			frame_profiling.call(computeIntersections, DIV_UP(size, BLOCK_SIZE), BLOCK_SIZE,
				i,
				dev_paths.subspan(0, num_paths),
				dev_geoms,
				dev_inters,
				dev_mesh_info,
				dev_cached_inters,
				*dev_tree
			);

			checkCUDAError(std::string("trace one bounce, inters size = " +
				std::to_string(MAX_INTERSECTION_TEST_SIZE)).c_str());
			cudaDeviceSynchronize();
		}

#ifdef DENOISE
		// initialize position and normal buffers for denoising
		// NOTE: must do this before the material sorting
		if (!depth && !iter) {
			denoise_buffers.set(dev_inters, dev_mesh_info.materials);
		}
#endif

		// stop before the shading stage if we're currently displaying a debug texture
		if (texture_debug_active) {
			return iter;
		}

		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by evaluating the BSDF.
#ifdef SORT_MAT
		thrust::sort_by_key(thrust::device, dev_inters, dev_inters + num_paths, dev_paths.get());
#endif

#ifdef FAKE_SHADE
#define shadeMaterial shadeFakeMaterial
#endif
		frame_profiling.call(shadeMaterial, DIV_UP(num_paths, BLOCK_SIZE), BLOCK_SIZE,
			iter,
			dev_paths.subspan(0, num_paths),
			dev_lights,
			dev_inters,
			dev_mesh_info.materials
		);
		checkCUDAError("shadeMaterial");
		cudaDeviceSynchronize();

#ifdef COMPACTION
		frame_profiling.begin();
		{
			num_paths = thrust::partition(thrust::device, dev_paths.get(), dev_paths.get() + num_paths, PathSegment::PartitionRule()) - dev_paths.get();
		}
		frame_profiling.end();
#endif // COMPACTION

#ifdef MAX_DEPTH_OVERRIDE
		if (depth == MAX_DEPTH_OVERRIDE)
			break;
#endif
	}
	
	// Assemble this iteration and apply it to the image
	frame_profiling.call(finalGather, DIV_UP(pixelcount, BLOCK_SIZE), BLOCK_SIZE,
		pixelcount, dev_image, dev_paths
	);
	cudaDeviceSynchronize();
	++cur_iter;

	// ----- write raytraced image to PBO ------
	// denoise
	if (enable_denoise) {
		frame_profiling.begin();
		{
			thrust::transform(
				thrust::device,
				dev_image.get(),
				dev_image.get() + pixelcount,
				denoise_image,
				RadianceToNormalizedRGB(cur_iter));

			ProfileHelper denoise_profiling("denoise");
			denoise_profiling.begin();
			{
				Denoiser::denoise(denoise_image, denoise_buffers, *denoise_params);
			}
			denoise_profiling.end();

			thrust::transform(
				thrust::device,
				denoise_image,
				denoise_image + pixelcount,
				s_pbo_dptr,
				NormalizedRGBToRGBA()
			);
		}
		frame_profiling.end();
	} else {
		frame_profiling.begin();
		{
			thrust::transform(
				thrust::device,
				dev_image.get(),
				dev_image.get() + pixelcount,
				s_pbo_dptr,
				RadianceToRGBA(cur_iter));
		}
		frame_profiling.end();
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
	return *dev_tree;
}
uchar4 const* PathTracer::getPBO() {
	return s_pbo_dptr;
}

void PathTracer::setDenoise(Denoiser::ParamDesc const& desc) {
	if (!denoise_params) {
		denoise_params = std::make_unique<Denoiser::ParamDesc>(desc);
	} else {
		*denoise_params = desc;
	}
}
#ifdef DENOISE_GBUF_OPTIMIZATION
__global__ void kern_visualize_pos(
	uchar4* pbo,
	glm::mat4x4 inv_view,
	glm::mat4x4 inv_proj,
	glm::ivec2 res,
	Denoiser::NormPos const* gbuf
) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int w = res[0], h = res[1];

	if (x >= w || y >= h)
		return;

	int idx = x + (y * w);
	pbo[idx] = Denoiser::DecodePosRGBA(inv_view, inv_proj, glm::ivec2(x, y), res)(gbuf[idx]);
}
#endif
void PathTracer::debugTexture(DebugTextureType type) {
	Camera const& cam = hst_scene->state.camera;
	int pixelcount = cam.resolution.x * cam.resolution.y;

	if (type == DebugTextureType::DIFFUSE_BUF) {
		auto tex = denoise_buffers.get_diffuse();
		thrust::transform(thrust::device, tex, tex + pixelcount, s_pbo_dptr, NormalizedRGBToRGBA());
	} else if (type == DebugTextureType::NORM_BUF) {
		auto tex = denoise_buffers.get_normal();
#ifdef DENOISE_GBUF_OPTIMIZATION
		thrust::transform(thrust::device, tex, tex + pixelcount, s_pbo_dptr, Denoiser::DecodeNormRGBA());
#else
		thrust::transform(thrust::device, tex, tex + pixelcount, s_pbo_dptr, NormalToRGBA());
#endif
	} else if (type == DebugTextureType::POS_BUF) {
		auto tex = denoise_buffers.get_pos();
#ifdef DENOISE_GBUF_OPTIMIZATION
		dim3 x(DIV_UP(cam.resolution.x, 8), DIV_UP(cam.resolution.y, 8));
		dim3 y(8, 8);
		kern_visualize_pos KERN_PARAM(x,y) (
			s_pbo_dptr,
			glm::inverse(CamState::get_view()),
			glm::inverse(CamState::get_proj()),
			cam.resolution,
			tex
		);
#else
		thrust::transform(thrust::device, tex, tex + pixelcount, s_pbo_dptr, 
			PosToRGBA(hst_scene->world_AABB.min(), hst_scene->world_AABB.max()));
#endif
	} else {
		texture_debug_active = false;
		return;
	}

	texture_debug_active = true;
}