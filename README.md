## <div align="center"> University of Pennsylvania, CIS 565: GPU Programming and Architecture </div>
# <div align="center"> CUDA Path Tracer </div>
- Name: Tongwei Dai
	- [LinkedIn Page](https://www.linkedin.com/in/tongwei-dai-583350177/)
- Tested on: Windows 10, i7-8700 @ 3.20 GHz 16GB, RTX 2070

## Overview
![](./img/Visual/cover.png)
- This is a basic path tracer written in C++ using the CUDA framework.
- Path tracing is a rendering method that simulates how the light scatters in the real world. Due to its physically accurate nature, many complex illumination phonomena come for free in path tracing.
- In path tracing, different physical materials are modeled with BSDF (Bidirectional Scattering Distribution Function). Each material would have such a function that describes how the light will bounce or transmit on its surface based on the directions of incoming and outgoing lights. 
	- In this project, I implemented some rudimentary BSDFs for ideal diffuse, reflective, refractive surfaces. I also attempted to simulate rough surfaces with microfacet models.
- Although a simple and powerful algorithm, path tracing can be quite costly as it simulates thousands if not millions of rays bouncing around in a 3D space. Spatial data structures are often used to quickly perform ray-object intersection tests.
	- In this project, I implemented an octree to avoid unnecessary computation in sparse areas of the scene.

## Features
### Visual Features
- [x] Simple Diffuse
- [x] Perfect Reflection
- [x] Fresnel-Modulated Specular Reflection and Refraction
- [x] Microfacet Reflection
- [x] Transparency
- [x] Antialiasing
- [ ] Depth of Field

### Performance Improvement
- [x] AABB bounding box
- [x] Octree Spatial Partition
- [x] First Bounce Caching
- [x] Ray Parallelization using Stream Compaction
- [x] Sort by Material Type

### Other
- [x] Checkpointing (Pause & Save to render later)
- [x] Arbitrary .obj file Loading
- [x] Diffuse, Normal Texture Mapping, Per-face Material
- [x] Octree/AABB Visualization

## 3rd Party Libraries Used
- [dear-imgui](https://github.com/ocornut/imgui): for graphical user interface
- [tiny-objloader](https://github.com/tinyobjloader/tinyobjloader): for loading and parsing .obj mesh files and .mtl material files
- [color-console](https://github.com/aafulei/color-console): for coloring error and warning messages in the console

## 3D Models Used
- all 3rd Party models are downloaded from [free3D](https://free3d.com/3d-models/obj) under the **Personal Use License**
	- [Humvee Vehicle](https://free3d.com/3d-model/humvee-vehicle-49947.html)
	- [Sofa](https://free3d.com/3d-model/sofa-801691.html)
	- [Toy Truck](https://free3d.com/3d-model/toy-truck-481161.html)

## Physically-based Rendering
- I heavily referenced [Physically Based Rendering: From Theory to Implementation](https://pbrt.org/) when writing the shading code.
- The cover image is actually a full picture of all the BSDFs I have implemented so far, as shown.
![](./img/Visual/shading_illustration.png)

### Microfacet Reflection
- Microfacet distribution can be thought of as a collection of surface normals. The rougher the surface, the greater the normal variation. When a ray is incident on the surface, a microfacet normal is sampled from that point. This irregularity gives us a rough-looking surface.
- When combined with reflection, a microfacet model is excellent in simulating an imperfect metallic surface, with a sharp highlight that gradually falls off.
- The roughness factor (r) controls how rough the surface is. Its effect is illustrated below.
![](./img/Visual/microfacet_comp_illus.png)

## Anti-aliasing
- Stochastic Anti-aliasing is used to reduce aliasing artifacts.
- The camera rays are randomly jittered within a pixel to introduce noises and therefore the "averaging" effect.
> the right picture shows the result of stochastic anti-aliasing
>
> ![](./img/AntiAliasing/comp.png)

## Mesh Loading and Texture Mapping
- currently only .obj files are supported
- materials of the mesh are stored in the .mtl of the same name, in the same directory

### Diffuse Texture Sampling
![](./img/Texture/test.png)

### Normal Mapping
- normal mapping is done by computing BTN matrices at each vertex and interpolating between them using barycentric coordinates
- the right hand side shows the object with a normal map in the following pictures

![](./img/NormalMap/comp1.png)
![](./img/NormalMap/comp2.png)

### Per-face Material
- a mesh may have multiple materials, so material-information is stored per face to allow for more freedom of adding textures and colors to a mesh
- in the pictures below, the toy truck has no UV and is textured by several diffuse materials, whereas the spider uses multiple diffuse textures.
![](./img/MeshLoading/truck.png)
![](./img/Texture/spider2.png)

## Performance Improvements
### AABB Culling
- Each object is assigned an AABB (Axis-Aligned Bounding Box)
- During the intersection test, the object's AABB will always be tested against the ray first.
- This is especially useful in scenes where there are lots of complex yet small meshes, because if a ray does not intersect with the AABB of a mesh (which happens a lot in such scenes), we can skip a great number of ray-triangle intersection tests.
- TODO: performance data

### Octree
- An octree can be viewed as a 3D version of a binary tree; it recursively and evenly divides a cube into 8 smaller cubes.
- We start with an AABB that encapsulates the entire scene as the root of the tree, and recursively subdivide it until either:
	- we have reached a pre-configured depth limit
	- or there is nothing but vacuum in the current AABB
- When we reach a leaf node, we store information about the objects it contains. Intermediate nodes do not contain object information.
- The octree is first constructed on the CPU and uploaded to the GPU where it is read-only.
	- any operations that modify its internal states are not supported on GPU.
- I have implemented a tool based on Imgui that helps with debugging and visualizing the octree, as shown below.
![](./img/Octree/demo.gif)
- However, an octree **does not** always outperform brute force.
	1. The depth limit of the octree is a crucial factor of performance. A poorly chosen depth limit will cause objects to be stored for multiple times in a lot of leaves. In the gif above, there are only 29,156 triangles in the scene, but an octree of depth 5 will contain 465,142 pieces of information in its leaf nodes! 
	2. The worst case scenario for the octree is a scene where there is a single small mesh and nothing else, as shown here. In this case, it may be faster to simply loop through all triangles.
		![](./img/Octree/worst.png)

- TODO: performance data

## Restartable Ray Tracing (Saving & Loading)
- Being able to pause and save the rendering progress is always a nice thing to have, especially when the scene takes hours to converge.
- Below is a short demo of how to use the saving & loading function.
![](./img/Checkpointing/demo.gif)