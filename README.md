CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Di Lu
  * [LinkedIn](https://www.linkedin.com/in/di-lu-0503251a2/)
  * [personal website](https://www.dluisnothere.com/)
* Tested on: Windows 11, i7-12700H @ 2.30GHz 32GB, NVIDIA GeForce RTX 3050 Ti

## Introduction

In 3D rendering, Pathtracing is a technique that generates realistic looking scenes/images by simulating light ray bounces. For this project, I implemented a CUDA path tracer for the GPU. In order to get the least noisy final output, 5000 calls to pathtrace are made whenever the camera is moved. The result of all 5000 pathtrace calls are then averaged to produce the final output. For each call to pathtrace, the light rays in the scene will bounce a maximum of 8 times.

For this pathtracer, we parallelize operations by Rays (AKA Path Segments), and made sure to sync all threads before moving on to the next parallel operation.

Overall, this project is a continuation of learning how to write CUDA kernel functions, optimize performance by adding memory coalescence, and very simple acceleration structures. The second part of the project introduced me to using TinyObjLoader, CudaTextureObjects, and various rendering techniques to get specific visual effects.

![](img/main2.png)

1. [Core Features](https://github.com/dluisnothere/Project3-CUDA-Path-Tracer#core-features): 
*  Simple Diffuse, Specular, and Imperfect Specular BSDF shading
*  Path continuation/termination using stream compaction
*  Continugous arrangement of materials based on materialId
*  First-bounce-cache for a specific camera angle.

2. [Additional Features](https://github.com/dluisnothere/Project3-CUDA-Path-Tracer#additional-features)
*  Refractive materials using Schlick's approximation
*  Depth of Field
*  Direct Lighting
*  Stochastic-sampled antialiasing
*  Arbitrary Mesh Loader with TinyObjLoader
*  UV Texturing, Procedural Texturing, and Bump Mapping

## Scene File Description

The scene files used in this project are laid out as blocks of text in this order: Materials, Textures (if any), and Objects in the scene. Each Object has a description of its translation, scale, and rotation, as well as which material it's using, and if it's a basic shape (such as sphere or cube), then that is also specified. If not a basic shape, then it specifies a path to its obj. If the Object also has a texture, then it will refer to the Id of the texture. Using Ids to keep track of scene attributes prevent over-copying of shared data between Objects.

## Core Features
**Shading kernel with BSDF Evaluation for Diffuse and Perfect/Imperfect Specular.**

Each Object in the scene has a reference to its material. Each material has attributes representing Reflection, Refraction, Specular, Index of Refraction, Color, and Emittance. For each ray, once it has gotten information about the object it has hit, I use a random number generate to generate a float between 0 and 1. Using this number, each ray will behave in a probabilistic manner and will either reflect, refract, or completely diffuse randomly in a cosine-weighted hemisphere (for lambertian reflection) based on the Reflection and Refraction material values in the scene file. 

**Path continuation/termination using Stream Compaction**

In order to avoid processing any unnecessary information, all paths which do not hit any of the Objects in the scene are removed from the list of active paths. This is done through the thrust library's built in stream-compaction function. Stream Compaction takes an array, runs a predicate on each element, and returns another list containing all the elements for which the predicate is true. 

**Contiguous arrangement of materials based on materialId**

For scenes which have a lot of different materials, sorting materials based on materialId makes sense to make reading from global memory faster.

**First bounce caching**

Since we restart rendering the scene whenever the camera moves, we know that the first bounce for each call to pathtrace for one particular camera angle will always be constant. We know exactly which object our first rays will hit and where. This way, we can avoid re-computing the first intersections for each of the 5000 calls to pathtrace. 

**Results of Core Features**

_All Materials Diffuse_            |  _Specular Sphere_ | _Imperfect Specular_
:-------------------------:|:-------------------------:|:--------------------:
![](img/part1Final.png) |  ![](img/part1FinalSpecular.png) | ![](img/imperfectSpecular.png)

## Additional Features
### Refractive Material

My implementation for refractive materials comes from [Physically Based Rendering](https://www.pbr-book.org/). If a ray decides it will refract, then I first get the material properties of material 1 and material 2 (in most cases, material 1 is air, which means it has an IOR of 1.003). Then, I obtain the incidence angle of my ray and the normal of the intersection point. Using this information, I can figure out if the angle is a critical angle or not. If it's a critical angle, then the light ray will simply bounce off the surface. Otherwise, it will refract. If the ray is inside the Object and a critical angle is hit, tjhen there will be total internal reflection.

This is the result of my refraction implementation:

_View 1_            |  _View 2 without specular sphere_ 
:-------------------------:|:-------------------------:
![](img/Refractive.png) |  ![](img/cornell.2022-09-28_22-10-45z.5000samp.png)

### Depth of Field

In order to obtain a depth of field effect, I referenced this online article: [Depth of Field in Path Tracing](https://medium.com/@elope139/depth-of-field-in-path-tracing-e61180417027#:~:text=Implementing%20depth%20of%20field%20in,out%20of%20focus%20will%20appear). Essentially, we determine two variables: FOCAL_LENGTH and APERTURE. FOCAL_LENGTH specifies how far away your focus point is (where the object becomes sharp). APERTURE is a proxy for how blurry everything out of focus should be. Once we know where the focal point P is in the scene, we can blur the rest of the scene by shifting our ray's origin and then recalculate direction such that it starts at the new origin and goes towards the focal point. These are my results:

_No Depth of Field_            |  _Depth of Field_ 
:-------------------------:|:-------------------------:
![](img/noDepthOfField.png) |  ![](img/depthFieldFinal.png)

### Stochastic Sampled Anti-Aliasing

_No Antialiasing_            |  _Antialiasing_ 
:-------------------------:|:-------------------------:
![](img/noAntialiasing.png) |  ![](img/antialiasing5000samp.png)

_No Antialiasing Closeup_            |  _Antialiasing Closeup_ 
:-------------------------:|:-------------------------:
![](img/noAntialiasing.png) |  ![](img/antialiasing5000samp.png)

### Direct Lighting

_No Direct Lighting_            |  _Direct Lighting_ 
:-------------------------:|:-------------------------:
![](img/NoDirectLighting.png) |  ![](img/DirectLighting.png)

### Arbitrary Mesh Loading with TinyObjLoader

![](img/refractiveKitty.png)

### UV Texture, Procedural Texture and Bump Mapping

![](img/completeMario2.png)

![](img/myLink.png)

![](img/nobump.png)

![](img/yesBumpmap.png)

![](img/procedural.png)

## Performance Analysis

- Stream compaction helps most after a few bounces. Print and plot the effects of stream compaction within a single iteration (i.e. the number of unterminated rays after each bounce) and evaluate the benefits you get from stream compaction.

- Compare scenes which are open (like the given cornell box) and closed (i.e. no light can escape the scene). Again, compare the performance effects of stream compaction! Remember, stream compaction only affects rays which terminate, so what might you expect?

My hypothesis is that since the scene is closed, all rays will hit something on each bounce. This means stream compaction won't remove any rays unless the ray ran out of bounces or it hit a light source.

- For optimizations that target specific kernels, we recommend using stacked bar graphs to convey total execution time and improvements in individual kernels. For example:

## Bloopers! :)

Too much antialiasing: me without my contacts

![](img/antialiasing1.png)

Cursed Black Hole Wahoo

![](img/wtfWahoo.png)
