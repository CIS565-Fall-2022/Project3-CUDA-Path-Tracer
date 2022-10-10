#include <cstdio>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
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
//#define CACHE_FIRST_BOUNCE
//#define SORT_BY_MATERIAL
//#define STREAM_COMPACT
#define ANTI_ALIASING

#define BLOCK_SIZE_1D 128
#define BLOCK_SIZE_2D 16

#define MIN_INTERSECT_DIST 0.0001f
#define MAX_INTERSECT_DIST 10000.0f

#define ENABLE_RECTS
#define ENABLE_SPHERES
#define ENABLE_TRIS
#define ENABLE_SQUAREPLANES

#define ENABLE_BVH_ACCEL


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



static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Tri* dev_tris = NULL;
static Light* dev_lights = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static BVHNode_GPU* dev_bvh_nodes = NULL;

static MISLightRay* dev_direct_light_rays = NULL;
static MISLightIntersection* dev_direct_light_isects = NULL;

static MISLightRay* dev_bsdf_light_rays = NULL;
static MISLightIntersection* dev_bsdf_light_isects = NULL;



thrust::device_ptr<PathSegment> thrust_dv_paths;
thrust::device_ptr<PathSegment> thrust_dv_paths_end;
thrust::device_ptr<ShadeableIntersection> thrust_dv_isects;

#ifdef CACHE_FIRST_BOUNCE
static ShadeableIntersection* dev_first_bounce_cache = NULL;
thrust::device_ptr<ShadeableIntersection> thrust_dv_first_bounce_cache;
#endif


static glm::vec3* dev_sample_colors = NULL;

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

	cudaMalloc(&dev_tris, scene->num_tris * sizeof(Tri));
	cudaMemcpy(dev_tris, scene->mesh_tris_sorted.data(), scene->num_tris * sizeof(Tri), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_bvh_nodes, scene->bvh_nodes_gpu.size() * sizeof(BVHNode_GPU));
	cudaMemcpy(dev_bvh_nodes, scene->bvh_nodes_gpu.data(), scene->bvh_nodes_gpu.size() * sizeof(BVHNode_GPU), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));
	cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Light), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));


	// FOR LIGHT SAMPLED MIS RAY
	cudaMalloc(&dev_direct_light_rays, pixelcount * sizeof(MISLightRay));

	cudaMalloc(&dev_direct_light_isects, pixelcount * sizeof(MISLightIntersection));
	cudaMemset(dev_direct_light_isects, 0, pixelcount * sizeof(MISLightIntersection));

	// FOR BSDF SAMPLED MIS RAY
	cudaMalloc(&dev_bsdf_light_rays, pixelcount * sizeof(MISLightRay));

	cudaMalloc(&dev_bsdf_light_isects, pixelcount * sizeof(MISLightIntersection));
	cudaMemset(dev_bsdf_light_isects, 0, pixelcount * sizeof(MISLightIntersection));

	// TODO: initialize any extra device memeory you need
#ifdef CACHE_FIRST_BOUNCE
	cudaMalloc(&dev_first_bounce_cache, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_first_bounce_cache, 0, pixelcount * sizeof(ShadeableIntersection));
#endif

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_tris);
	cudaFree(dev_bvh_nodes);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_lights);
	cudaFree(dev_direct_light_rays);
	cudaFree(dev_direct_light_isects);
	cudaFree(dev_bsdf_light_rays);
	cudaFree(dev_bsdf_light_isects);


#ifdef CACHE_FIRST_BOUNCE
	cudaFree(dev_first_bounce_cache);
#endif

	checkCUDAError("pathtraceFree");
}

#ifdef ANTI_ALIASING
// AA
__global__ void generateRayFromThinLensCamera(Camera cam, int iter, int traceDepth, float jitterX, float jitterY, glm::vec3 thinLensCamOrigin, glm::vec3 newRef,
	PathSegment* pathSegments)
{
	__shared__ PathSegment mat[BLOCK_SIZE_2D][BLOCK_SIZE_2D];

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);

	if (x < cam.resolution.x && y < cam.resolution.y) {
		mat[threadIdx.x][threadIdx.y] = pathSegments[index];
		PathSegment& segment = mat[threadIdx.x][threadIdx.y];

		segment.ray.origin = thinLensCamOrigin;
		segment.rayThroughput = glm::vec3(1.0f, 1.0f, 1.0f);
		segment.accumulatedIrradiance = glm::vec3(0.0f, 0.0f, 0.0f);
		segment.prev_hit_was_specular = false;

		float jittered_x = ((float)x) + jitterX;
		float jittered_y = ((float)y) + jitterY;

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(
			glm::normalize(newRef - thinLensCamOrigin) - cam.right * cam.pixelLength.x * (jittered_x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * (jittered_y - (float)cam.resolution.y * 0.5f)
		);

		segment.ray.direction_inv = 1.0f / glm::vec3(segment.ray.direction);
		segment.ray.ray_dir_sign[0] = segment.ray.direction_inv.x < 0.0f;
		segment.ray.ray_dir_sign[1] = segment.ray.direction_inv.y < 0.0f;
		segment.ray.ray_dir_sign[2] = segment.ray.direction_inv.z < 0.0f;

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;

		pathSegments[index] = mat[threadIdx.x][threadIdx.y];
	}
}

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, float jitterX, float jitterY,
	PathSegment* pathSegments)
{
	__shared__ PathSegment mat[BLOCK_SIZE_2D][BLOCK_SIZE_2D];

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);

	if (x < cam.resolution.x && y < cam.resolution.y) {
		mat[threadIdx.x][threadIdx.y] = pathSegments[index];
		PathSegment& segment = mat[threadIdx.x][threadIdx.y];

		segment.ray.origin = cam.position;
		segment.rayThroughput = glm::vec3(1.0f, 1.0f, 1.0f);
		segment.accumulatedIrradiance = glm::vec3(0.0f, 0.0f, 0.0f);
		segment.prev_hit_was_specular = false;

		float jittered_x = ((float)x) + jitterX;
		float jittered_y = ((float)y) + jitterY;

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(
			cam.view - cam.right * cam.pixelLength.x * (jittered_x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * (jittered_y - (float)cam.resolution.y * 0.5f)
		);

		segment.ray.direction_inv = 1.0f / glm::vec3(segment.ray.direction);
		segment.ray.ray_dir_sign[0] = segment.ray.direction_inv.x < 0.0f;
		segment.ray.ray_dir_sign[1] = segment.ray.direction_inv.y < 0.0f;
		segment.ray.ray_dir_sign[2] = segment.ray.direction_inv.z < 0.0f;

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;

		pathSegments[index] = mat[threadIdx.x][threadIdx.y];
	}
}

#else
// NO AA
__global__ void generateRayFromCamera(Camera cam, int traceDepth, PathSegment* pathSegments)
{

	__shared__ PathSegment mat[BLOCK_SIZE_2D][BLOCK_SIZE_2D];
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);

	if (x < cam.resolution.x && y < cam.resolution.y) {
		mat[threadIdx.x][threadIdx.y] = pathSegments[index];
		PathSegment& segment = mat[threadIdx.x][threadIdx.y];

		segment.ray.origin = cam.position;
		segment.rayThroughput = glm::vec3(1.0f, 1.0f, 1.0f);
		segment.accumulatedIrradiance = glm::vec3(0.0f, 0.0f, 0.0f);
		segment.prev_hit_was_specular = false;

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);

		segment.ray.direction_inv = 1.0f / glm::vec3(segment.ray.direction);
		segment.ray.ray_dir_sign[0] = segment.ray.direction_inv.x < 0.0f;
		segment.ray.ray_dir_sign[1] = segment.ray.direction_inv.y < 0.0f;
		segment.ray.ray_dir_sign[2] = segment.ray.direction_inv.z < 0.0f;

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;

		pathSegments[index] = mat[threadIdx.x][threadIdx.y];
	}
}
#endif

__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, Tri* tris
	, int tris_size
	, ShadeableIntersection* intersections
	, BVHNode_GPU* bvh_nodes
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
#ifndef STREAM_COMPACT
		if (pathSegments[path_index].remainingBounces == 0) {
			return;
		}
#endif
		Ray r = pathSegments[path_index].ray;

		ShadeableIntersection isect;
		isect.t = MAX_INTERSECT_DIST;

		float t;


		glm::vec3 tmp_normal;
		int obj_ID = -1;

#ifdef ENABLE_TRIS
		if (tris_size != 0) {

#ifdef ENABLE_BVH_ACCEL
			int stack_pointer = 0;
			int cur_node_index = 0;
			int node_stack[32];
			BVHNode_GPU cur_node;
			glm::vec3 P;
			glm::vec3 s;
			float t1;
			float t2;
			float tmin;
			float tmax;
			while (true) {
				cur_node = bvh_nodes[cur_node_index];

				// (ray-aabb test node)
				t1 = (cur_node.AABB_min.x - r.origin.x) * r.direction_inv.x;
				t2 = (cur_node.AABB_max.x - r.origin.x) * r.direction_inv.x;

				tmin = glm::min(t1, t2);
				tmax = glm::max(t1, t2);

				t1 = (cur_node.AABB_min.y - r.origin.y) * r.direction_inv.y;
				t2 = (cur_node.AABB_max.y - r.origin.y) * r.direction_inv.y;

				tmin = glm::max(tmin, glm::min(t1, t2));
				tmax = glm::min(tmax, glm::max(t1, t2));

				t1 = (cur_node.AABB_min.z - r.origin.z) * r.direction_inv.z;
				t2 = (cur_node.AABB_max.z - r.origin.z) * r.direction_inv.z;

				tmin = glm::max(tmin, glm::min(t1, t2));
				tmax = glm::min(tmax, glm::max(t1, t2));

				if (tmax >= tmin) {
					// we intersected AABB
					if (cur_node.tri_index != -1) {
						// this is leaf node
						// triangle intersection test
						Tri tri = tris[cur_node.tri_index];

						t = glm::dot(tri.plane_normal, (tri.p0 - r.origin)) / glm::dot(tri.plane_normal, r.direction);
						if (t >= -0.0001f) {
							P = r.origin + t * r.direction;

							// barycentric coords
							s = glm::vec3(glm::length(glm::cross(P - tri.p1, P - tri.p2)),
								glm::length(glm::cross(P - tri.p2, P - tri.p0)),
								glm::length(glm::cross(P - tri.p0, P - tri.p1))) / tri.S;

							if (s.x >= -0.0001f && s.x <= 1.0001f && s.y >= -0.0001f && s.y <= 1.0001f &&
								s.z >= -0.0001f && s.z <= 1.0001f && (s.x + s.y + s.z <= 1.0001f) && (s.x + s.y + s.z >= -0.0001f) && isect.t > t) {
								isect.t = t;
								isect.materialId = tri.mat_ID;
								isect.surfaceNormal = glm::normalize(s.x * tri.n0 + s.y * tri.n1 + s.z * tri.n2);
							}
						}
						// if last node in tree, we are done
						if (stack_pointer == 0) {
							break;
						}
						// otherwise need to check rest of the things in the stack
						stack_pointer--;
						cur_node_index = node_stack[stack_pointer];
					}
					else {	
						node_stack[stack_pointer] = cur_node.offset_to_second_child;
						stack_pointer++;
						cur_node_index++;
					}
				}
				else {
					// didn't intersect AABB, remove from stack
					if (stack_pointer == 0) {
						break;
					}
					stack_pointer--;
					cur_node_index = node_stack[stack_pointer];
				}
			}

#else
			for (int j = 0; j < tris_size; ++j)
			{
				// triangle intersection test
				Tri tri = tris[j];

				t = glm::dot(tri.plane_normal, (tri.p0 - r.origin)) / glm::dot(tri.plane_normal, r.direction);
				if (t < 0.0f) continue;

				glm::vec3 P = r.origin + t * r.direction;

				// barycentric coords
				glm::vec3 s = glm::vec3(glm::length(glm::cross(P - tri.p1, P - tri.p2)),
					glm::length(glm::cross(P - tri.p2, P - tri.p0)),
					glm::length(glm::cross(P - tri.p0, P - tri.p1))) / tri.S;

				if (s.x >= -0.0001f && s.x <= 1.0001f && s.y >= -0.0001f && s.y <= 1.0001f &&
					s.z >= -0.0001f && s.z <= 1.0001f && (s.x + s.y + s.z <= 1.0001f) && (s.x + s.y + s.z >= -0.0001f) && isect.t > t) {
					isect.t = t;
					isect.materialId = tri.mat_ID;
					isect.surfaceNormal = glm::normalize(s.x * tri.n0 + s.y * tri.n1 + s.z * tri.n2);
				}
			}
#endif

	}
#endif


		for (int i = 0; i < geoms_size; ++i)
		{
			Geom& geom = geoms[i];



			if (geom.type == SPHERE) {
#ifdef ENABLE_SPHERES
				t = sphereIntersectionTest(geom, r, tmp_normal);
#endif                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
			}
			else if (geom.type == SQUAREPLANE) {
#ifdef ENABLE_SQUAREPLANES
				t = squareplaneIntersectionTest(geom, r, tmp_normal);
#endif	
			}
			else {
#ifdef ENABLE_RECTS
			t = boxIntersectionTest(geom, r, tmp_normal);
#endif
			}

			if (depth == 0 && glm::dot(tmp_normal, r.direction) > 0.0) { 
				continue; 
			}
			else if (isect.t > t) {
				isect.t = t;
				isect.materialId = geom.materialid;
				isect.surfaceNormal = tmp_normal;
			}
			
		}

		if (isect.t >= MAX_INTERSECT_DIST) {
			// hits nothing
			pathSegments[path_index].remainingBounces = 0;
		}
		else {
			intersections[path_index] = isect;
		}
	}
}

__global__ void genMISRaysKernel(
	int iter
	, int num_paths
	, int max_depth
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, MISLightRay* direct_light_rays
	, MISLightRay* bsdf_light_rays
	, Light* lights
	, int num_lights
	, Geom* geoms
	, MISLightIntersection* direct_light_isects
	, MISLightIntersection* bsdf_light_isects
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		if (pathSegments[idx].remainingBounces == 0) {
			return;
		}

		ShadeableIntersection intersection = shadeableIntersections[idx];
		Material material = materials[intersection.materialId];
		
		if (material.emittance > 0.0f) {
			if (pathSegments[idx].remainingBounces == max_depth || pathSegments[idx].prev_hit_was_specular) {
				// only color lights on first hit
				pathSegments[idx].accumulatedIrradiance += (material.R * material.emittance) * pathSegments[idx].rayThroughput;
			}
			pathSegments[idx].remainingBounces = 0;
			return;
		}

		pathSegments[idx].prev_hit_was_specular = material.type == SPEC_BRDF || material.type == SPEC_BTDF || material.type == SPEC_GLASS || material.type == SPEC_PLASTIC;

		if (pathSegments[idx].prev_hit_was_specular) {
			return;
		}

		glm::vec3 intersect_point = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;

		thrust::default_random_engine rng = makeSeededRandomEngine(iter + glm::abs(intersect_point.x) + idx, iter + glm::abs(intersect_point.y) + pathSegments[idx].remainingBounces, pathSegments[idx].remainingBounces + glm::abs(intersect_point.z));

		thrust::uniform_real_distribution<float> u01(0, 1);

		// choose light to directly sample
		direct_light_rays[idx].light_ID = bsdf_light_rays[idx].light_ID = lights[glm::min((int)(glm::floor(u01(rng) * (float)num_lights)), num_lights - 1)].geom_ID;

		Geom& light = geoms[direct_light_rays[idx].light_ID];

		Material& light_material = materials[light.materialid];

		////////////////////////////////////////////////////
		// LIGHT SAMPLED
		////////////////////////////////////////////////////

		// generate light sampled wi
		glm::vec3 wi = glm::vec3(0.0f);
		float absDot = 0.0f;
		glm::vec3 f = glm::vec3(0.0f);
		float pdf_L = 0.0f;
		float pdf_B = 0.0f;

		if (light.type == SQUAREPLANE) {
			glm::vec2 p_obj_space = glm::vec2(u01(rng) - 0.5f, u01(rng) - 0.5f);
			glm::vec3 p_world_space = glm::vec3(light.transform * glm::vec4(p_obj_space.x, p_obj_space.y, 0.0f, 1.0f));
			wi = glm::normalize(glm::vec3(p_world_space - intersect_point));
			absDot = glm::dot(wi, glm::normalize(glm::vec3(light.invTranspose * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f))));
			
			if (absDot < 0.0001f) {
				absDot = glm::abs(absDot);
				// pdf of square plane light = distanceSq / (absDot * lightArea)
				float dist = glm::length(p_world_space - intersect_point);
				if (absDot > 0.0001f) {
					pdf_L = (dist * dist) / (absDot * light.scale.x * light.scale.y);
				}
			}
			else {
				pdf_L = 0.0f;
			}
		}

		direct_light_rays[idx].ray.origin = intersect_point + (wi * 0.001f);
		direct_light_rays[idx].ray.direction = wi;
		direct_light_rays[idx].ray.direction_inv = 1.0f / wi;
		direct_light_rays[idx].ray.ray_dir_sign[0] = wi.x < 0.0f;
		direct_light_rays[idx].ray.ray_dir_sign[1] = wi.y < 0.0f;
		direct_light_rays[idx].ray.ray_dir_sign[2] = wi.z < 0.0f;
		

		absDot = glm::abs(glm::dot(intersection.surfaceNormal, wi));
		// generate f, pdf, absdot from light sampled wi
		if (material.type == SPEC_BRDF) {
			// spec refl
			direct_light_rays[idx].f = glm::vec3(0.0f);
		}
		else if (material.type == SPEC_BTDF) {
			// spec refr
			direct_light_rays[idx].f = glm::vec3(0.0f);
		}
		else if (material.type == SPEC_GLASS) {
			// spec glass
			direct_light_rays[idx].f = glm::vec3(0.0f);
		}
		else if (material.type == SPEC_PLASTIC) {
			pdf_B = absDot * 0.31831f / 2.0f;
			f = material.R * 0.31831f;
		}
		else {
			pdf_B = absDot * 0.31831f;
			f = material.R * 0.31831f; // INV_PI
			 
		}
		direct_light_rays[idx].f = f;
		direct_light_rays[idx].pdf = pdf_B;

		// LTE = f * Li * absDot / pdf
		if (pdf_L <= 0.0001f) {
			direct_light_isects[idx].LTE = glm::vec3(0.0f, 0.0f, 0.0f);
		}
		else {
			direct_light_isects[idx].LTE = light_material.emittance * light_material.R * f * absDot / pdf_L;

		}

		// MIS Power Heuristic
		if (pdf_L <= 0.0001f && pdf_B <= 0.0001f) {
			direct_light_isects[idx].w = 0.0f;
		}
		else {
			direct_light_isects[idx].w = (pdf_L * pdf_L) / ((pdf_L * pdf_L) + (pdf_B * pdf_B));
		}


		////////////////////////////////////////////////////
		// BSDF SAMPLED
		////////////////////////////////////////////////////

		if (material.type == SPEC_BRDF) {
			// spec refl
			wi = glm::reflect(pathSegments[idx].ray.direction, intersection.surfaceNormal);
			absDot = glm::abs(glm::dot(intersection.surfaceNormal, wi));
			pdf_B = 1.0f;
			if (absDot == 0.0f) {
				f = material.R;
			}
			else {
				f = material.R / absDot;
			}
		}
		else if (material.type == SPEC_BTDF) {
			// spec refr
			float eta = material.ior;
			if (glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction) < 0.0f) {
				// outside
				eta = 1.0f / eta;
				wi = glm::refract(pathSegments[idx].ray.direction, intersection.surfaceNormal, eta);
			}
			else {
				// inside
				wi = glm::refract(pathSegments[idx].ray.direction, -intersection.surfaceNormal, eta);
			}
			absDot = glm::abs(glm::dot(intersection.surfaceNormal, wi));
			pdf_B = 1.0f;
			if (glm::length(wi) <= 0.0001f) {
				// total internal reflection
				f = glm::vec3(0.0f);
			}
			if (absDot == 0.0f) {
				f = material.T;
			}
			else {
				f = material.T / absDot;
			}
		}
		else if (material.type == SPEC_GLASS) {
			// spec glass
			float eta = material.ior;
			if (u01(rng) < 0.5f) {
				// spec refl
				wi = glm::reflect(pathSegments[idx].ray.direction, intersection.surfaceNormal);
				absDot = glm::abs(glm::dot(intersection.surfaceNormal, wi));
				pdf_B = 1.0f;
				if (absDot == 0.0f) {
					f = material.R;
				}
				else {
					f = material.R / absDot;
				}
				f *= fresnelDielectric(glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction), material.ior);
			}
			else {
				// spec refr
				if (glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction) < 0.0f) {
					// outside
					eta = 1.0f / eta;
					wi = glm::refract(pathSegments[idx].ray.direction, intersection.surfaceNormal, eta);
				}
				else {
					// inside
					wi = glm::refract(pathSegments[idx].ray.direction, -intersection.surfaceNormal, eta);
				}
				absDot = glm::abs(glm::dot(intersection.surfaceNormal, wi));
				pdf_B = 1.0f;
				if (glm::length(wi) <= 0.0001f) {
					// total internal reflection
					f = glm::vec3(0.0f);
				}
				else if (absDot == 0.0f) {
					f = material.T;
				}
				else {
					f = material.T / absDot;
				}
				f *= glm::vec3(1.0f) - fresnelDielectric(glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction), material.ior);
			}
			f *= 2.0f;
		}
		else if (material.type == SPEC_PLASTIC) {
			// spec glass
			if (u01(rng) < 0.5f) {
				// diffuse
				wi = glm::normalize(calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng, u01));
				absDot = glm::abs(glm::dot(intersection.surfaceNormal, wi));
				pdf_B = absDot * 0.31831f;
				f = material.R * 0.31831f; // INV_PI
				f *= glm::vec3(1.0f) - fresnelDielectric(glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction), material.ior);
			}
			else {
				// spec refl
				wi = glm::reflect(pathSegments[idx].ray.direction, intersection.surfaceNormal);
				absDot = glm::abs(glm::dot(intersection.surfaceNormal, wi));
				pdf_B = 1.0f;
				if (absDot == 0.0f) {
					f = material.T;
				}
				else {
					f = material.T / absDot;
				}
				f *= fresnelDielectric(glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction), material.ior);
			}
			f *= 2.0f;
		}
		else {
			// diffuse
			wi = glm::normalize(calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng, u01));
			absDot = glm::abs(glm::dot(intersection.surfaceNormal, wi));
			pdf_B = absDot * 0.31831f;
			f = material.R * 0.31831f; // INV_PI
		}


		// Change ray direction
		bsdf_light_rays[idx].ray.origin = intersect_point + (wi * 0.001f);
		bsdf_light_rays[idx].ray.direction = wi;
		bsdf_light_rays[idx].ray.direction_inv = 1.0f / wi;
		bsdf_light_rays[idx].ray.ray_dir_sign[0] = wi.x < 0.0f;
		bsdf_light_rays[idx].ray.ray_dir_sign[1] = wi.y < 0.0f;
		bsdf_light_rays[idx].ray.ray_dir_sign[2] = wi.z < 0.0f;
		bsdf_light_rays[idx].f = f;


		// LTE = f * Li * absDot / pdf
		absDot = glm::abs(glm::dot(intersection.surfaceNormal, bsdf_light_rays[idx].ray.direction));
		bsdf_light_rays[idx].pdf = pdf_B;

		if (pdf_B <= 0.0001f) {
			bsdf_light_isects[idx].LTE = glm::vec3(0.0f, 0.0f, 0.0f);
		}
		else {
			bsdf_light_isects[idx].LTE = light_material.emittance * light_material.R * bsdf_light_rays[idx].f * absDot / pdf_B;
		}
		
	}
}

__global__ void computeDirectLightIsects(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, MISLightRay* direct_light_rays
	, Geom* geoms
	, int geoms_size
	, Tri* tris
	, int tris_size
	, MISLightIntersection* direct_light_intersections
	, BVHNode_GPU* bvh_nodes
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{

		if (pathSegments[path_index].remainingBounces == 0) {
			return;
		}
		else if (pathSegments[path_index].prev_hit_was_specular) {
			return;
		}

		MISLightRay r = direct_light_rays[path_index];

		float t_min = MAX_INTERSECT_DIST;
		int obj_ID = -1;


		float t;

		glm::vec3 tmp_normal;

#ifdef ENABLE_TRIS
		if (tris_size != 0) {
#ifdef ENABLE_BVH_ACCEL
			int stack_pointer = 0;
			int cur_node_index = 0;
			int node_stack[32];
			BVHNode_GPU cur_node;
			glm::vec3 P;
			glm::vec3 s;
			float t1;
			float t2;
			float tmin;
			float tmax;
			while (true) {
				cur_node = bvh_nodes[cur_node_index];

				// (ray-aabb test node)
				t1 = (cur_node.AABB_min.x - r.ray.origin.x) * r.ray.direction_inv.x;
				t2 = (cur_node.AABB_max.x - r.ray.origin.x) * r.ray.direction_inv.x;

				tmin = glm::min(t1, t2);
				tmax = glm::max(t1, t2);

				t1 = (cur_node.AABB_min.y - r.ray.origin.y) * r.ray.direction_inv.y;
				t2 = (cur_node.AABB_max.y - r.ray.origin.y) * r.ray.direction_inv.y;

				tmin = glm::max(tmin, glm::min(t1, t2));
				tmax = glm::min(tmax, glm::max(t1, t2));

				t1 = (cur_node.AABB_min.z - r.ray.origin.z) * r.ray.direction_inv.z;
				t2 = (cur_node.AABB_max.z - r.ray.origin.z) * r.ray.direction_inv.z;

				tmin = glm::max(tmin, glm::min(t1, t2));
				tmax = glm::min(tmax, glm::max(t1, t2));

				if (tmax >= tmin) {
					// we intersected AABB
					if (cur_node.tri_index != -1) {
						// this is leaf node
						// triangle intersection test
						Tri tri = tris[cur_node.tri_index];

						t = glm::dot(tri.plane_normal, (tri.p0 - r.ray.origin)) / glm::dot(tri.plane_normal, r.ray.direction);
						if (t >= -0.0001f) {
							P = r.ray.origin + t * r.ray.direction;

							// barycentric coords
							s = glm::vec3(glm::length(glm::cross(P - tri.p1, P - tri.p2)),
								glm::length(glm::cross(P - tri.p2, P - tri.p0)),
								glm::length(glm::cross(P - tri.p0, P - tri.p1))) / tri.S;

							if (s.x >= -0.0001f && s.x <= 1.0001f && s.y >= -0.0001f && s.y <= 1.0001f &&
								s.z >= -0.0001f && s.z <= 1.0001f && (s.x + s.y + s.z <= 1.0001f) && (s.x + s.y + s.z >= -0.0001f) && t_min > t) {
								t_min = t;
							}
						}
						// if last node in tree, we are done
						if (stack_pointer == 0) {
							break;
						}
						// otherwise need to check rest of the things in the stack
						stack_pointer--;
						cur_node_index = node_stack[stack_pointer];
					}
					else {
						node_stack[stack_pointer] = cur_node.offset_to_second_child;
						stack_pointer++;
						cur_node_index++;
					}
				}
				else {
					// didn't intersect AABB, remove from stack
					if (stack_pointer == 0) {
						break;
					}
					stack_pointer--;
					cur_node_index = node_stack[stack_pointer];
				}
			}

#else
			for (int j = 0; j < tris_size; ++j)
			{
				// triangle intersection test
				Tri tri = tris[j];

				t = glm::dot(tri.plane_normal, (tri.p0 - r.ray.origin)) / glm::dot(tri.plane_normal, r.ray.direction);
				if (t < 0.0f) continue;

				glm::vec3 P = r.ray.origin + t * r.ray.direction;

				// barycentric coords
				glm::vec3 s = glm::vec3(glm::length(glm::cross(P - tri.p1, P - tri.p2)),
					glm::length(glm::cross(P - tri.p2, P - tri.p0)),
					glm::length(glm::cross(P - tri.p0, P - tri.p1))) / tri.S;

				if (s.x >= -0.0001f && s.x <= 1.0001f && s.y >= -0.0001f && s.y <= 1.0001f &&
					s.z >= -0.0001f && s.z <= 1.0001f && (s.x + s.y + s.z <= 1.0001f) && (s.x + s.y + s.z >= -0.0001f) && t_min > t) {
					t_min = t;
				}
			}
#endif
	}
#endif

		for (int i = 0; i < geoms_size; ++i)
		{
			Geom& geom = geoms[i];



			if (geom.type == SPHERE) {
#ifdef ENABLE_SPHERES
				t = sphereIntersectionTest(geom, r.ray, tmp_normal);
#endif
			}
			else if (geom.type == SQUAREPLANE) {
#ifdef ENABLE_SQUAREPLANES
				t = squareplaneIntersectionTest(geom, r.ray, tmp_normal);
#endif
			}
			else {
#ifdef ENABLE_RECTS
				t = boxIntersectionTest(geom, r.ray, tmp_normal);
#endif
			}

			if (t_min > t)
			{
				t_min = t;
				obj_ID = i;
			}
		}

		if (obj_ID != r.light_ID) {
			direct_light_intersections[path_index].LTE = glm::vec3(0.0f, 0.0f, 0.0f);
			direct_light_intersections[path_index].w = 0.0f;
		}

		// LTE = f * Li * absDot / pdf
		// Already have f, Li, absDot, and pdf from when we generated ray
		// MIS Power Heuristic already calulated in raygen
	}
}

__global__ void computeBSDFLightIsects(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, MISLightRay* bsdf_light_rays
	, Geom* geoms
	, int geoms_size
	, Tri* tris
	, int tris_size
	, MISLightIntersection* bsdf_light_intersections
	, BVHNode_GPU* bvh_nodes
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{

		if (pathSegments[path_index].remainingBounces == 0) {
			return;
		}
		else if (pathSegments[path_index].prev_hit_was_specular) {
			return;
		}

		MISLightRay r = bsdf_light_rays[path_index];

		float t_min = MAX_INTERSECT_DIST;
		int obj_ID = -1;
		float pdf_L_B = 0.0f;
		float t;
		glm::vec3 hit_normal;

		glm::vec3 tmp_normal;

#ifdef ENABLE_TRIS
		if (tris_size != 0) {
#ifdef ENABLE_BVH_ACCEL
			int stack_pointer = 0;
			int cur_node_index = 0;
			int node_stack[32];
			BVHNode_GPU cur_node;
			glm::vec3 P;
			glm::vec3 s;
			float t1;
			float t2;
			float tmin;
			float tmax;
			while (true) {
				cur_node = bvh_nodes[cur_node_index];

				// (ray-aabb test node)
				t1 = (cur_node.AABB_min.x - r.ray.origin.x) * r.ray.direction_inv.x;
				t2 = (cur_node.AABB_max.x - r.ray.origin.x) * r.ray.direction_inv.x;

				tmin = glm::min(t1, t2);
				tmax = glm::max(t1, t2);

				t1 = (cur_node.AABB_min.y - r.ray.origin.y) * r.ray.direction_inv.y;
				t2 = (cur_node.AABB_max.y - r.ray.origin.y) * r.ray.direction_inv.y;

				tmin = glm::max(tmin, glm::min(t1, t2));
				tmax = glm::min(tmax, glm::max(t1, t2));

				t1 = (cur_node.AABB_min.z - r.ray.origin.z) * r.ray.direction_inv.z;
				t2 = (cur_node.AABB_max.z - r.ray.origin.z) * r.ray.direction_inv.z;

				tmin = glm::max(tmin, glm::min(t1, t2));
				tmax = glm::min(tmax, glm::max(t1, t2));

				if (tmax >= tmin) {
					// we intersected AABB
					if (cur_node.tri_index != -1) {
						// this is leaf node
						// triangle intersection test
						Tri tri = tris[cur_node.tri_index];

						t = glm::dot(tri.plane_normal, (tri.p0 - r.ray.origin)) / glm::dot(tri.plane_normal, r.ray.direction);
						if (t >= -0.0001f) {
							P = r.ray.origin + t * r.ray.direction;

							// barycentric coords
							s = glm::vec3(glm::length(glm::cross(P - tri.p1, P - tri.p2)),
								glm::length(glm::cross(P - tri.p2, P - tri.p0)),
								glm::length(glm::cross(P - tri.p0, P - tri.p1))) / tri.S;

							if (s.x >= -0.0001f && s.x <= 1.0001f && s.y >= -0.0001f && s.y <= 1.0001f &&
								s.z >= -0.0001f && s.z <= 1.0001f && (s.x + s.y + s.z <= 1.0001f) && (s.x + s.y + s.z >= -0.0001f) && t_min > t) {
								t_min = t;
								hit_normal = glm::normalize(s.x * tri.n0 + s.y * tri.n1 + s.z * tri.n2);
							}
						}
						// if last node in tree, we are done
						if (stack_pointer == 0) {
							break;
						}
						// otherwise need to check rest of the things in the stack
						stack_pointer--;
						cur_node_index = node_stack[stack_pointer];
					}
					else {
						node_stack[stack_pointer] = cur_node.offset_to_second_child;
						stack_pointer++;
						cur_node_index++;
					}
				}
				else {
					// didn't intersect AABB, remove from stack
					if (stack_pointer == 0) {
						break;
					}
					stack_pointer--;
					cur_node_index = node_stack[stack_pointer];
				}
			}

#else
			for (int j = 0; j < tris_size; ++j)
			{
				// triangle intersection test
				Tri tri = tris[j];

				t= glm::dot(tri.plane_normal, (tri.p0 - r.ray.origin)) / glm::dot(tri.plane_normal, r.ray.direction);
				if (t < 0.0f) continue;

				glm::vec3 P = r.ray.origin + t * r.ray.direction;

				// barycentric coords
				glm::vec3 s = glm::vec3(glm::length(glm::cross(P - tri.p1, P - tri.p2)),
					glm::length(glm::cross(P - tri.p2, P - tri.p0)),
					glm::length(glm::cross(P - tri.p0, P - tri.p1))) / tri.S;

				if (s.x >= -0.0001f && s.x <= 1.0001f && s.y >= -0.0001f && s.y <= 1.0001f &&
					s.z >= -0.0001f && s.z <= 1.0001f && (s.x + s.y + s.z <= 1.0001f) && (s.x + s.y + s.z >= -0.0001f) && t_min > t) {
					t_min = t;
					hit_normal = glm::normalize(s.x * tri.n0 + s.y * tri.n1 + s.z * tri.n2);
				}
			}
#endif
		}
#endif
		

		for (int i = 0; i < geoms_size; ++i)
		{
			Geom& geom = geoms[i];



			if (geom.type == SPHERE) {
#ifdef ENABLE_SPHERES
				t = sphereIntersectionTest(geom, r.ray, tmp_normal);
#endif
			}
			else if (geom.type == SQUAREPLANE) {
#ifdef ENABLE_SQUAREPLANES
				t = squareplaneIntersectionTest(geom, r.ray, tmp_normal);
#endif
			}
			else {
#ifdef ENABLE_RECTS
			t = boxIntersectionTest(geom, r.ray, tmp_normal);
#endif
		}

			if (t_min > t)
			{
				hit_normal = tmp_normal;
				t_min = t;
				obj_ID = i;
			}
		}

		float absDot = glm::dot(hit_normal, r.ray.direction);

		if (obj_ID == r.light_ID && absDot < 0.0f) {

			absDot = glm::abs(absDot);
			pdf_L_B = (t_min * t_min) / (absDot * geoms[obj_ID].scale.x * geoms[obj_ID].scale.y);

			// LTE = f * Li * absDot / pdf
			// Already have f, Li, and pdf from when we generated ray
			bsdf_light_intersections[path_index].LTE *= absDot;

			// MIS Power Heuristic
			if (pdf_L_B == 0.0f && r.pdf == 0.0f) {
				bsdf_light_intersections[path_index].w = 0.0f;
			}
			else {
				bsdf_light_intersections[path_index].w = (r.pdf * r.pdf) / ((r.pdf * r.pdf) + (pdf_L_B * pdf_L_B));
			}
		}
		else {
			bsdf_light_intersections[path_index].LTE = glm::vec3(0.0f, 0.0f, 0.0f);
			bsdf_light_intersections[path_index].w = 0.0f;
		}
	}
}

__global__ void shadeMaterialUberKernel(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, MISLightIntersection* direct_light_isects
	, MISLightIntersection* bsdf_light_isects
	, int num_lights
	, PathSegment* pathSegments
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		if (pathSegments[idx].remainingBounces == 0) {
			return;
		}
		ShadeableIntersection intersection = shadeableIntersections[idx];
		MISLightIntersection direct_light_intersection = direct_light_isects[idx];
		MISLightIntersection bsdf_light_intersection = bsdf_light_isects[idx];

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);

		Material material = materials[intersection.materialId];

		glm::vec3 intersect_point = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;

		// Combine direct light and bsdf light samples with Power Heuristic
		if (!pathSegments[idx].prev_hit_was_specular) {
			pathSegments[idx].accumulatedIrradiance += pathSegments[idx].rayThroughput * (float)num_lights *
				(direct_light_intersection.w * direct_light_intersection.LTE +
					bsdf_light_intersection.w * bsdf_light_intersection.LTE);
		}


		// GI LTE
		scatterRay(pathSegments[idx], intersect_point,
			intersection.surfaceNormal,
			material,
			rng);
		pathSegments[idx].remainingBounces--;
	}
}

__global__ void russianRouletteKernel(int iter, int num_paths, PathSegment* pathSegments)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < num_paths)
	{
		if (pathSegments[idx].remainingBounces == 0) {
			return;
		}
		thrust::default_random_engine rng = makeSeededRandomEngine(iter + idx, idx, pathSegments[idx].remainingBounces + idx);
		thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
		float random_num = u01(rng);
		float max_channel = glm::max(glm::max(pathSegments[idx].rayThroughput.r, pathSegments[idx].rayThroughput.g), pathSegments[idx].rayThroughput.b);
		if (max_channel < random_num) {
			pathSegments[idx].remainingBounces = 0;
		}
		else {
			pathSegments[idx].rayThroughput /= max_channel;
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
		image[iterationPath.pixelIndex] += iterationPath.accumulatedIrradiance;
	}
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		pix /= iter;

		// reinhard (HDR)
		pix /= (pix + glm::vec3(1.0f));

		// gamma correction
		pix = glm::pow(pix, glm::vec3(0.454545f));

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}


struct is_done
{
	__host__ __device__
		bool operator()(const PathSegment &path)
	{
		return path.remainingBounces != 0;
	}
};

struct material_sort
{
	__host__ __device__
		bool operator()(const ShadeableIntersection& isect_0, const ShadeableIntersection& isect_1)
	{
		return isect_0.materialId < isect_1.materialId;
	}
};

#ifdef CACHE_FIRST_BOUNCE
void cacheFirstBounce(int iter, int cur_paths, dim3 &numblocksPathSegmentTracing, 
	const int blockSize1d, PerformanceTimer &perf_timer) {

	// clean shading chunks
	perf_timer.startGpuTimer();
	cudaMemset(dev_first_bounce_cache, 0, cur_paths * sizeof(ShadeableIntersection));
	perf_timer.endGpuTimer();
	//std::cout << "cudaMemset: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;

	// tracing
	perf_timer.startGpuTimer();
	computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
		0
		, cur_paths
		, dev_paths
		, dev_geoms
		, hst_scene->geoms.size()
		, dev_tris
		, hst_scene->num_tris
		, dev_first_bounce_cache
		, dev_bvh_nodes
		);
	checkCUDAError("trace cached intersections");
	cudaDeviceSynchronize();
	perf_timer.endGpuTimer();
	//std::cout << "computeIntersections: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;

}



void useCachedFirstBounce(int iter, int traceDepth, int &cur_paths, int &depth, bool &iterationComplete,
	PathSegment* &dev_path_end, dim3& numblocksPathSegmentTracing,
	const int blockSize1d, PerformanceTimer& perf_timer ) {

	perf_timer.startGpuTimer();
	cudaMemcpy(dev_intersections, dev_first_bounce_cache,
		cur_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
	perf_timer.endGpuTimer();
	//std::cout << "copy cache to intersections: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;
	depth++;

#ifdef SORT_BY_MATERIAL
	perf_timer.startGpuTimer();
	thrust::stable_sort_by_key(thrust::device, thrust_dv_isects, thrust_dv_isects + cur_paths, thrust_dv_paths, material_sort());
	perf_timer.endGpuTimer();
	//std::cout << "sort by material: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;
#endif

	perf_timer.startGpuTimer();
	genMISRaysKernel << <numblocksPathSegmentTracing, blockSize1d >> > (
		iter,
		cur_paths,
		traceDepth,
		dev_intersections,
		dev_paths,
		dev_materials,
		dev_direct_light_rays,
		dev_bsdf_light_rays,
		dev_lights,
		hst_scene->lights.size(),
		dev_geoms,
		dev_direct_light_isects,
		dev_bsdf_light_isects
		);
	checkCUDAError("gen MIS rays (light sampled and bsdf sampled)");
	cudaDeviceSynchronize();
	perf_timer.endGpuTimer();
	//std::cout << "genMISRaysKernel: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;


	perf_timer.startGpuTimer();
	computeDirectLightIsects << <numblocksPathSegmentTracing, blockSize1d >> > (
		depth
		, cur_paths
		, dev_paths
		, dev_direct_light_rays
		, dev_geoms
		, hst_scene->geoms.size()
		, dev_tris
		, hst_scene->num_tris
		, dev_direct_light_isects
		, dev_bvh_nodes
		);
	checkCUDAError("get direct lighting intersections");
	cudaDeviceSynchronize();
	perf_timer.endGpuTimer();
	//std::cout << "computeIntersections: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;

	perf_timer.startGpuTimer();
	computeBSDFLightIsects << <numblocksPathSegmentTracing, blockSize1d >> > (
		depth
		, cur_paths
		, dev_paths
		, dev_bsdf_light_rays
		, dev_geoms
		, hst_scene->geoms.size()
		, dev_tris
		, hst_scene->num_tris
		, dev_bsdf_light_isects
		, dev_bvh_nodes
		);
	checkCUDAError("get bsdf lighting intersections");
	cudaDeviceSynchronize();
	perf_timer.endGpuTimer();
	//std::cout << "computeIntersections: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;

	perf_timer.startGpuTimer();
	shadeMaterialUberKernel << <numblocksPathSegmentTracing, blockSize1d >> > (
		iter,
		cur_paths,
		dev_intersections,
		dev_direct_light_isects,
		dev_bsdf_light_isects,
		hst_scene->lights.size(),
		dev_paths,
		dev_materials
		);
	checkCUDAError("shade one bounce");
	cudaDeviceSynchronize();
	perf_timer.endGpuTimer();
	//std::cout << "shadeMaterialUberKernel: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;

#ifdef STREAM_COMPACT
	perf_timer.startGpuTimer();
	thrust_dv_paths_end = thrust::stable_partition(thrust::device, thrust_dv_paths, thrust_dv_paths + cur_paths, is_done());
	cudaDeviceSynchronize();
	perf_timer.endGpuTimer();
	//std::cout << "stream compaction: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;
	dev_path_end = thrust_dv_paths_end.get();
	cur_paths = dev_path_end - dev_paths;
#endif

	if (depth == traceDepth || cur_paths == 0) { iterationComplete = true; }

	if (guiData != NULL)
	{
		guiData->TracedDepth = depth;
	}
}

#endif

void pathtrace(uchar4* pbo, int frame, int iter) {
	PerformanceTimer perf_timer;
	perf_timer.startCpuTimer();
	
	//std::cout << "============================== " << iter << " ==============================" << std::endl;

	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D,
		(cam.resolution.y + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D);


	// 1D block for path tracing
	const int blockSize1d = BLOCK_SIZE_1D;

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	thrust_dv_paths = thrust::device_pointer_cast<PathSegment>(dev_paths);
	thrust_dv_paths_end = thrust::device_pointer_cast<PathSegment>(dev_path_end);
	thrust_dv_isects = thrust::device_pointer_cast<ShadeableIntersection>(dev_intersections);

	bool iterationComplete = false;
	int cur_paths = num_paths;

	dim3 numblocksPathSegmentTracing = (cur_paths + blockSize1d - 1) / blockSize1d;

	perf_timer.endCpuTimer();

#ifdef CACHE_FIRST_BOUNCE
	thrust_dv_first_bounce_cache = thrust::device_pointer_cast<ShadeableIntersection>(dev_first_bounce_cache);

	perf_timer.startGpuTimer();

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, traceDepth, dev_paths);

	checkCUDAError("generate camera ray");
	perf_timer.endGpuTimer();

	if (iter == 1) {
		// handle first bounce (depth == 0)
		cacheFirstBounce(iter, cur_paths, numblocksPathSegmentTracing, blockSize1d, perf_timer);

	}
	// compute depth = 0 using the cached first bounce intersections
	useCachedFirstBounce(iter, traceDepth, cur_paths, depth, iterationComplete, dev_path_end,
		numblocksPathSegmentTracing, blockSize1d, perf_timer);
#else

	// gen ray
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, iter, iter);
	thrust::uniform_real_distribution<float> upixel(0.0, 1.0f);

	float jitterX = upixel(rng);
	float jitterY = upixel(rng);

	if (cam.lens_radius > 0.0f) {
		// thin lens camera model based on my implementation from CIS 561
		// also based on https://www.semanticscholar.org/paper/A-Low-Distortion-Map-Between-Disk-and-Square-Shirley-Chiu/43226a3916a85025acbb3a58c17f6dc0756b35ac?p2df
		glm::mat3 M = glm::mat3(cam.right, cam.up, cam.view);

		float focalT = (cam.focal_distance / glm::length(cam.lookAt - cam.position));
		glm::vec3 newRef = cam.position + focalT * (cam.lookAt - cam.position);
		glm::vec2 thinLensSample = glm::vec2(upixel(rng), upixel(rng));

		// turn square shaped random sample domain into disc shaped
		glm::vec3 warped = glm::vec3(0.0f);
		glm::vec2 sampleRemap = 2.0f * thinLensSample - glm::vec2(1.0f);
		float r, theta = 0.0f;
		if (glm::abs(sampleRemap.x) > glm::abs(sampleRemap.y)) {
			r = sampleRemap.x;
			theta = (PI / 4.0f) * (sampleRemap.y / sampleRemap.x);
		}
		else {
			r = sampleRemap.y;
			theta = (PI / 2.0f) - (PI / 4.0f) * (sampleRemap.x / sampleRemap.y);
		}
		warped = r * glm::vec3(glm::cos(theta), glm::sin(theta), 0.0f);

		glm::vec3 lensPoint = cam.lens_radius * warped;

		glm::vec3 thinLensCamOrigin = cam.position + M * lensPoint;

		
		//std::cout << "sample init: " << perf_timer.getCpuElapsedTimeForPreviousOperation() << std::endl;

		perf_timer.startGpuTimer();

		generateRayFromThinLensCamera << <blocksPerGrid2d, blockSize2d >> > (cam,
			iter, traceDepth, jitterX, jitterY, thinLensCamOrigin, newRef, dev_paths);

		checkCUDAError("generate camera ray");
		perf_timer.endGpuTimer();
	}
	else {
		perf_timer.startGpuTimer();

		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam,
			iter, traceDepth, jitterX, jitterY, dev_paths);

		checkCUDAError("generate camera ray");
		perf_timer.endGpuTimer();
	}

#endif

	while (!iterationComplete) {

		// clean shading chunks
		//cudaMemset(dev_intersections, 0, cur_paths * sizeof(ShadeableIntersection));

		// tracing
		numblocksPathSegmentTracing = (cur_paths + blockSize1d - 1) / blockSize1d;
		perf_timer.startGpuTimer();
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, cur_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_tris
			, hst_scene->num_tris
			, dev_intersections
			, dev_bvh_nodes
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		perf_timer.endGpuTimer();
		//std::cout << "computeIntersections: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;		
		depth++;

#ifdef SORT_BY_MATERIAL
		perf_timer.startGpuTimer();
		thrust::stable_sort_by_key(thrust::device, thrust_dv_isects, thrust_dv_isects + cur_paths, thrust_dv_paths, material_sort());
		perf_timer.endGpuTimer();
		//std::cout << "sort by material: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;
#endif

		perf_timer.startGpuTimer();
		genMISRaysKernel << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			cur_paths,
			traceDepth,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_direct_light_rays,
			dev_bsdf_light_rays,
			dev_lights,
			hst_scene->lights.size(),
			dev_geoms,
			dev_direct_light_isects,
			dev_bsdf_light_isects
			);
		checkCUDAError("gen MIS rays (light sampled and bsdf sampled)");
		cudaDeviceSynchronize();
		perf_timer.endGpuTimer();
		//std::cout << "genMISRaysKernel: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;


		perf_timer.startGpuTimer();
		computeDirectLightIsects << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, cur_paths
			, dev_paths
			, dev_direct_light_rays
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_tris
			, hst_scene->num_tris
			, dev_direct_light_isects
			, dev_bvh_nodes
			);
		checkCUDAError("get direct lighting intersections");
		cudaDeviceSynchronize();
		perf_timer.endGpuTimer();
		//std::cout << "computeIntersections: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;

		perf_timer.startGpuTimer();
		computeBSDFLightIsects << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, cur_paths
			, dev_paths
			, dev_bsdf_light_rays
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_tris
			, hst_scene->num_tris
			, dev_bsdf_light_isects
			, dev_bvh_nodes
			);
		checkCUDAError("get bsdf lighting intersections");
		cudaDeviceSynchronize();
		perf_timer.endGpuTimer();
		//std::cout << "computeIntersections: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;

		perf_timer.startGpuTimer();
		shadeMaterialUberKernel << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			cur_paths,
			dev_intersections,
			dev_direct_light_isects,
			dev_bsdf_light_isects,
			hst_scene->lights.size(),
			dev_paths,
			dev_materials
			);
		checkCUDAError("shade one bounce");
		cudaDeviceSynchronize();
		perf_timer.endGpuTimer();
		//std::cout << "shadeMaterialUberKernel: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;

		// RUSSIAN ROULETTE
		if (depth >= 4) {
			perf_timer.startGpuTimer();
			russianRouletteKernel << <numblocksPathSegmentTracing, blockSize1d >> > (
				iter,
				cur_paths,
				dev_paths
				);
			checkCUDAError("shade one bounce");
			cudaDeviceSynchronize();
			perf_timer.endGpuTimer();
			//std::cout << "russianRouletteKernel: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;
		}


#ifdef STREAM_COMPACT
		perf_timer.startGpuTimer();
		thrust_dv_paths_end = thrust::stable_partition(thrust::device, thrust_dv_paths, thrust_dv_paths + cur_paths, is_done());
		cudaDeviceSynchronize();
		perf_timer.endGpuTimer();
		//std::cout << "stream compaction: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;
		dev_path_end = thrust_dv_paths_end.get();
		cur_paths = dev_path_end - dev_paths;
#endif

		if (depth == traceDepth || cur_paths == 0) { iterationComplete = true; }

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	perf_timer.startGpuTimer();
	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);
	perf_timer.endGpuTimer();
	//std::cout << "finalGather: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;
	
	//if ((iter & 64) >> 6 || iter < 2) {

		perf_timer.startGpuTimer();
		// Send results to OpenGL buffer for rendering
		sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
		perf_timer.endGpuTimer();
		//std::cout << "sendImageToPBO: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;

		// Retrieve image from GPU
		perf_timer.startGpuTimer();
		cudaMemcpy(hst_scene->state.image.data(), dev_image,
			pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
		perf_timer.endGpuTimer();
		//std::cout << "last cudamemcpy: " << perf_timer.getGpuElapsedTimeForPreviousOperation() << std::endl;
		checkCUDAError("pathtrace");
	//}
}