#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
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

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

//Personal Trigger(use to do performance analysis)
#define ENABLE_FIRST_INTERSECTION_CACHE 1
#define ENABLE_RAY_SORTING 1

#define ENABLE_DEPTH_OF_FIELD 1



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

static Triangle* dev_triangles = NULL;

static PathSegment* dev_paths = NULL;
//Added
static PathSegment* dev_final_paths = NULL;

static ShadeableIntersection* dev_intersections = NULL;

// TODO: static variables for device memory, any extra info you need, etc
// ...
static thrust::device_ptr<PathSegment> thrust_dev_paths;
static thrust::device_ptr<ShadeableIntersection> thrust_dev_intersections;
static int* dev_materialID = NULL;
static thrust::device_ptr<int> thrust_dev_materialID;
static int* dev_materialID_copy = NULL;
static thrust::device_ptr<int> thrust_dev_materialIDCpy;

//Texture Data
static cudaTextureObject_t* dev_texObjs=NULL;
static std::vector<cudaArray_t> dev_texArray;
static std::vector<cudaTextureObject_t> texObjs;

//Mesh data for GPU
static PrimitiveData dev_prim_data;

static Mesh* dev_meshes = NULL;

#if ENABLE_FIRST_INTERSECTION_CACHE
static ShadeableIntersection* dev_first_intersect;
bool bounceAlreadyCached = false;
#endif


void textureInit(const Texture& tex,int i)
{
	//Allocate CUDA Array
	//From NVIDIA Document
	cudaTextureObject_t texObj;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
	//allocate texture memory
	cudaMallocArray(&dev_texArray[i],&channelDesc,tex.width,tex.height);

	//Set pitch of sourece (the width in memory in bytes of the 2D array pointed to src)
	//don't have padding const size_t
    // Copy texture image in host memory to device memory
	cudaMemcpyToArray(dev_texArray[i],0,0,tex.image,tex.width*tex.height*tex.component*sizeof(unsigned char),cudaMemcpyHostToDevice);

	//specify texture parameters
	struct cudaResourceDesc resDesc;
	memset(&resDesc,0,sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = dev_texArray[i];

	//specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 1;

	//create texture object
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	cudaMemcpy(dev_texObjs+i,&texObj,sizeof(cudaTextureObject_t),cudaMemcpyHostToDevice);

	texObjs.push_back(texObj);
}
void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}


template <class T>
void mallocAndCopytoGPU(T*& d, std::vector<T>& h) {
	cudaMalloc(&d, h.size() * sizeof(T));
	cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
}



void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	dev_final_paths = dev_paths;



	mallocAndCopytoGPU<Material>(dev_materials,scene->materials);
	mallocAndCopytoGPU<Geom>(dev_geoms, scene->geoms);
	mallocAndCopytoGPU<Mesh>(dev_meshes, scene->meshes);

	//cudaMalloc(&dev_triangles, scene->mesh_triangles.size() * sizeof(Triangle));
	//cudaMemcpy(dev_triangles, scene->mesh_triangles.data(), sizeof(Triangle) * scene->mesh_triangles.size(), cudaMemcpyHostToDevice);

	mallocAndCopytoGPU<Primitive>(dev_prim_data.primitives, scene->primitives);
	mallocAndCopytoGPU<uint16_t>(dev_prim_data.indices, scene->mesh_indices);
	mallocAndCopytoGPU<glm::vec3>(dev_prim_data.normals, scene->mesh_normals);
	mallocAndCopytoGPU<glm::vec2>(dev_prim_data.texCoords, scene->mesh_uvs);
	mallocAndCopytoGPU<glm::vec3>(dev_prim_data.vertices, scene->mesh_vertices);
	mallocAndCopytoGPU<glm::vec4>(dev_prim_data.tangents, scene->mesh_tangents);

	//Meshes

	//Texture memory
	texObjs.clear();
	dev_texArray.clear();
	cudaMalloc(&dev_texObjs,scene->textures.size()*sizeof(cudaTextureObject_t));
	dev_texArray.resize(scene->textures.size());

	for (int i = 0; i < scene->textures.size(); i++)
	{
		textureInit(scene->textures[i], i);
	}


	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	thrust_dev_intersections = thrust::device_pointer_cast(dev_intersections);

	// TODO: initialize any extra device memeory you need

	thrust_dev_paths = thrust::device_pointer_cast(dev_paths);

	cudaMalloc(&dev_materialID, pixelcount * sizeof(int));
	cudaMemset(dev_materialID, 0, pixelcount * sizeof(int));
	thrust_dev_materialID = thrust::device_pointer_cast(dev_materialID);

	cudaMalloc(&dev_materialID_copy, pixelcount * sizeof(int));
	cudaMemset(dev_materialID_copy, 0, pixelcount * sizeof(int));
	thrust_dev_materialIDCpy = thrust::device_pointer_cast(dev_materialID_copy);


#if ENABLE_FIRST_INTERSECTION_CACHE
	cudaMalloc(&dev_first_intersect, pixelcount * sizeof(ShadeableIntersection));

	cudaMemset(dev_first_intersect,0,pixelcount*sizeof(ShadeableIntersection));
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
	cudaFree(dev_materialID);
	cudaFree(dev_materialID_copy);
	//cudaFree(dev_final_paths);
	cudaFree(dev_triangles);
	dev_prim_data.free();


	for (int i = 0; i < texObjs.size(); i++) {
		cudaDestroyTextureObject(texObjs[i]);
		cudaFreeArray(dev_texArray[i]);
	}

#if ENABLE_FIRST_INTERSECTION_CACHE
	cudaFree(dev_first_intersect);
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

	bool enableDepthField = false;
	bool enableStochasticAA = true;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
		thrust::uniform_real_distribution<float> u01(0, 1);

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f);

		//SSAA anti-aliasing
		if (enableStochasticAA)
		{
			float dx = u01(rng) * cam.pixelLength.x - cam.pixelLength.x / 2;
			float dy = u01(rng) * cam.pixelLength.y - cam.pixelLength.y / 2;
			segment.ray.origin = cam.position;
			segment.ray.direction = glm::normalize(cam.view
				- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
				- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
				- cam.right * dx - cam.up * dy);
		}
		else
	    {
		    segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f));
	    }
		//DepthField
		if (enableDepthField)
		{
			float lensRadius = 0.3f; // 0.003f
			float focalDistance = 5.f;
			thrust::normal_distribution<float> n01(0, 1);
			float theta = u01(rng) * TWO_PI;
			glm::vec3 circlePerturb = lensRadius * n01(rng) * (cos(theta) * cam.right + sin(theta) * cam.up);
			glm::vec3 originalDir = segment.ray.direction;
			float ft = focalDistance / glm::dot(originalDir, cam.view);
			segment.ray.origin = cam.position + circlePerturb;
			segment.ray.direction = glm::normalize(ft * originalDir - circlePerturb);
		}
		
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
		// TODO: implement antialiasing by jittering the ray
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
	, Mesh* meshes
	, int geoms_size
	, PrimitiveData dev_prim_data
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

		glm::vec2 uv;
		glm::vec4 tangent;
		int materialID = -1;

		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;
	

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv;
		glm::vec4 tmp_tangent;
		int temp_materialID = -1;

		// naive parse through global geoms
	//	printf("Mesh data debug index 0 1 2: %d %d %d \n", dev_prim_data.indices[0], dev_prim_data.indices[1], dev_prim_data.indices[2]);

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				temp_materialID = geom.materialid;
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				temp_materialID = geom.materialid;
			}
			else if (geom.type == MESH)
			{

				t = meshIntersectionTest(geom, meshes[geom.mesh_id], dev_prim_data, pathSegment.ray,
					tmp_intersect, tmp_normal, tmp_uv, tmp_tangent,temp_materialID,pathSegment.color);
	
			//	std::cout << "Debug msg: mesh triggered" << std::endl;
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				materialID = temp_materialID;
				normal = tmp_normal;
				tangent = tmp_tangent;
				uv = tmp_uv;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}
		ShadeableIntersection& intersection = intersections[path_index];
		if (hit_geom_index == -1)
		{
			intersection.t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersection.t = t_min;
			intersection.materialId = materialID;
			intersection.surfaceNormal = normal;
			intersection.tangent = tangent;
			//Add here
			intersection.intersectionPoint = intersect_point;
			//printf("get intersections \n");
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



__global__ void BSDFShading(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, cudaTextureObject_t* textures
)
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < num_paths)
	{

		ShadeableIntersection intersection = shadeableIntersections[index];
		if (intersection.t > 0.0f)
		{
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			//glm::vec3 materialColor = glm::vec3(material.pbrVal.baseColor);
	


			//Ray ends when ray hit the light
			if (material.emittance>0.0)
			{
				pathSegments[index].color *= (material.color * material.emittance);
				pathSegments[index].remainingBounces = 0;
			}
			else
			{
				//need intersection position
				glm::vec3 inter = getPointOnRay(pathSegments[index].ray, intersection.t);
				scatterRay(pathSegments[index],intersection, inter, intersection.surfaceNormal, material,textures,rng);
				//Debug
				pathSegments[index].remainingBounces -= 1;
			}
		}
		else
		{
			pathSegments[index].color = glm::vec3(0.f);
			pathSegments[index].remainingBounces = 0;
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

struct not_termintated
{
	__host__ __device__
		bool operator()(const PathSegment& p)
	{
		return p.remainingBounces > 0;
	}
};
void pathtrace(uchar4* pbo, int frame, int iter,bool sortMaterial) {
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
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
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
	// 
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing

	int depth = 0;

	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = pixelcount;
	int triangleSize = hst_scene->mesh_triangles.size();
	

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	bool iterationComplete = false;
	ShadeableIntersection* intersections = NULL;

	while (!iterationComplete) 
	{
		    dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if	ENABLE_FIRST_INTERSECTION_CACHE
			if (depth == 0 && iter != 1)
			{
				intersections = dev_first_intersect;
			}
#endif
			if (intersections == NULL)
			{
				cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

			
				// tracing
				computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth
					, num_paths
					, dev_paths
					, dev_geoms
					, dev_meshes
					, hst_scene->geoms.size()
					, dev_prim_data
					, dev_intersections
					);
				checkCUDAError("trace one bounce");
				cudaDeviceSynchronize();
				//Since cached
#if ENABLE_FIRST_INTERSECTION_CACHE
				if (depth == 0 && iter == 1)
				{
					cudaMemcpy(dev_first_intersect, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
				}
#endif
				intersections = dev_intersections;
			}

			depth++;

#if ENABLE_FIRST_INTERSECTION_CACHE
			if (depth == 0 && iter == 1)
				intersections = dev_first_intersect;
#endif

	// TODO:
	// --- Shading Stage ---
	 // Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.
#if ENABLE_RAY_SORTING
		//sort the intersections
		thrust::sort_by_key(thrust::device,dev_intersections,dev_intersections+num_paths,dev_paths,compareIntersection());
#endif

	   BSDFShading << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			intersections,
			dev_paths,
			dev_materials,
		    dev_texObjs
			);
	   cudaDeviceSynchronize();

	   //String Compaction Here
	    dev_paths = thrust::stable_partition(thrust::device,dev_paths,dev_paths+num_paths,isPathCompleted());
		//num_paths was changed here
		num_paths = dev_path_end - dev_paths;
		iterationComplete = (num_paths == 0);
		intersections=NULL;
	
		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}	
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	//num_path already equal to zero
	//finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_final_paths);
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_final_paths);

	dev_paths = dev_final_paths;

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
