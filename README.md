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

Overall, this project is a continuation of learning how to write CUDA kernel functions, optimize performance by adding memory coalescence, and very simple acceleration structures. The second part of the project introduced me to using TinyObjLoader, CudaTextureObjects, and various rendering techniques to get specific types of images:

![](img/main.png)

![](img/mainAntialiasing.png)

1. [Core Features](https://github.com/dluisnothere/Project3-CUDA-Path-Tracer#core-features): 
*  Simple Diffuse, Specular, and Imperfect Specular BSDF shading
*  Path continuation/termination using stream compaction
*  Continugous arrangement of materials based on materialId
*  First-bounce-cache for a specific camera angle.

3. [Additional Features](https://github.com/dluisnothere/Project3-CUDA-Path-Tracer#additional-features)
*  Refractive materials
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

![](img/part1Final.png)

_All Materials Diffuse_

![](img/part1FinalSpecular.png)

_Specular Sphere_

![](img/imperfectSpecular.png)

_Imperfect specular floors reflecting three objects_

## Additional Features
### Refractive Material

![](img/Refractive.png)

### Depth of Field

![](img/noDepthOfField.png)

![](img/depthFieldFinal.png)

### Stochastic Sampled Anti-Aliasing

![](img/noAntialiasing.png)

![](img/antialiasing5000samp.png)

### Direct Lighting

![](img/NoDirectLighting.png)

![](img/DirectLighting.png)

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
