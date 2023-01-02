CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

![](img/cover-image.png)  
`motorcycle.txt, motorcycle.gltf`: 5000 samples, depth 8, 960 x 720 px

Constance Wang
  * [LinkedIn](https://www.linkedin.com/in/conswang/)

Tested on AORUS 15P XD laptop with specs:  
- Windows 11 22000.856  
- 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz 2.30 GHz  
- NVIDIA GeForce RTX 3070 Laptop GPU  

### Features
This is a Monte-Carlo pathtracer with GPU-accelerated intersection tests, shading, and path culling in CUDA.

- Core features
  - Diffuse and perfect specular shaders
  - Performance optimizations
    - Sorting rays by material
    - Path termination using stream compaction
    - Cache first bounce intersections
- Additional features
  - Gltf 2.0 loading & rendering
  - Texture mapping & bump mapping
  - Metallic shader
  - Bounding volume hierarchy
  - Stochastic sampled anti-aliasing

### Usage
The base code has been modified to take two arguments. The first argument is a filepath to the original txt scene format, and the second, optional argument is a filepath to a gltf file.

```
./pathtracer.exe [motorcycle.txt] [motorcycle.gltf]
```
#### Dependencies
- Clone and add [tinygltf.h](https://github.com/syoyo/tinygltf) to external includes

### Feature Toggles
All macros are defined in `sceneStructs.h`.  
- Performance
  - `SORT_BY_MATERIALS`
  - `BVH`: toggle bounding volume hierarchy
  - `CACHE_FIRST_BOUNCE`
- Visual
  - `ANTI_ALIAS`
  - `ROUGHNESS_METALLIC`: render metallic shader
- Debugging
  - `SHOW_NORMALS`: render normals as color
  - `SHOW_METALLIC`: render metallicness as color

### GLTF
Most arbitrary gltf files exported from Blender can be loaded and rendered without errors. The base code is used to render the lights and camera while gltf is used to load meshes.

- Scene graph traversal is supported
  - Both matrix and translation/rotation/scale attributes are supported to describe local transformations of nodes
  - See `motorcycle.gltf` for an example of a complex scene with many nodes in a tree-like structure
- Copies position, normal, tangent, UV, and index buffers into an interleaved array on the GPU
- Texture loading

### Texture Mapping

#### Color Map

#### Normal Map
Gltf normal textures must be in tangent space. They are transformed into world space using a TBN matrix. Intersection normals and tangents are interpolated from the vertex normal and tangent buffers from the file.

### Metallic Shader
I partially implemented gltf's microfacet (PBR metallic/roughness) workflow by adding a metallic shader. The metallic value from 0 to 1 comes from either the gltf material's `pbrMetallicRoughness.metallicFactor` or is read from a texture, where the blue channel is the metallic value. The metallic value is used to interpolate the diffuse and metallic shaders.

![](img/metallic_box.png)  
`box.txt, Box With Spaces.gltf`: 1000 samples, depth 8

By setting `#define SHOW_METALLIC 1`, we can debug the metallic value. Here is what it looks like for the motorcycle scene. Brighter blue means more metallic. On the motorcycle, the metallic factor is defined for the entire material, whereas on the vending machine, the metallic factor comes from a texture. 

![](img/metallic-debug.png)

### Bounding Volume Hierarchy

### Anti-Aliasing
Implemented anti-aliasing by jittering the camera ray in the up and right directions by the amount `boxSize`, aka. jitter ~ uniform(-boxSize/2, boxSize/2). This looks visually pleasing enough that it wasn't worth using a Gaussian distribution, since calculating its pdf would be much more expensive.

When anti-aliasing is ON, first bounce caching must be turned OFF.

| boxSize | Scene | Close-up |
|--------|------|-------|
| 0 (no AA) |![](img/antialias_cornell_avocado_0.png) | ![](img/aa-0-zoom.png) |
|1|![](img/antialias_cornell_avocado_1.png) | ![](img/aa-1-zoom.png)|
|2| ![](img/antialias_cornell_avocado_2.png) | ![](img/aa-2-zoom.png)