CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Chang Liu
  * [LinkedIn](https://www.linkedin.com/in/chang-liu-0451a6208/)
  * [Personal website](https://hummawhite.github.io/)
* Tested on personal laptop:
  - i7-12700 @ 4.90GHz with 16GB RAM
  - RTX 3070 Ti Laptop 8GB

## Overview

This is our third project of CIS 565 Fall 2022. This project is about implementing a path tracer running on GPU with CUDA programming library. 

## Representative Outcome

![](./img/photo_realistic.jpg)

<p align="center">A virtual camera is capturing another virtual camera</p>

<div align="center">
    <table>
    <tr>
        <th>Scene Specs</th>
        <th><a href="./scenes/pbr_texture.txt">[./scenes/pbr_texture.txt]</a></th>
    </tr>
    <tr>
        <td>Resolution</td>    
        <td>2400 x 1800</td> 
    </tr> 
    <tr> 
        <td>Samples Per Pixel</td> 
        <td>3000</td>
    </tr>
    <tr> 
        <td>Render Time</td> 
        <td>&lt; 7 minutes (&gt; 7.5 frames per second)</td>
    </tr>
    <tr> 
        <td>Million Rays Per Second</td> 
        <td>32.4</td>
    </tr>
    <tr> 
        <td>Triangle Count</td> 
        <td>25637</td>
    </tr>
</table>
</div>

![](./img/rungholt_2.jpg)

<p align="center">Large scene rendering (6,704,264 triangles)</p><br>

![](./img/dragon_dof_2.jpg)

<p align="center">Lens effect: depth of field with heart-shaped bokehs</p>



## Features

### Visual

#### Direct Lighting with Multiple Importance Sampling



#### Importance Sampled HDR Environment Map (Skybox)

Tired of repeatedly setting up and adjusting "virtual artificial" light sources for the best effect? Let's introduce some natural light from real world that can easily improve our rendering quality at no cost.



#### Physically-Based Materials

##### Lambertian Diffuse

##### Metallic Workflow: Expressive and Artist-Friendly

##### Dielectric

#### Normal Map & PBR Texture



#### Physically Based Camera

##### Depth of Field



| No DOF                      | DOF                     |
| --------------------------- | ----------------------- |
| ![](./img/aperture_off.jpg) | ![](./img/aperture.jpg) |

##### Custom Bokeh Shape

This is my favorite part of the project.

In DOF, we sample points on the camera's aperture disk to create blurred background and foreground. This idea can even be extended by stochastically sampling a mask image instead of the circle shape of the aperture:

<div align="center">
	<img src="./scenes/texture/star3.jpg" width="15%"/>
	<img src="./scenes/texture/heart2.jpg" width="15%"/>
</div>

| Star Mask                      | Heart Mask                    |
| ------------------------------ | ----------------------------- |
| ![](./img/aperture_custom.jpg) | ![](./img/aperture_heart.jpg) |

Which creates very interesting and amazing results.

The sampling method is the same as what is used to pick up 

##### Panorama (360 Degree Spherical)



![](./img/sponza_pano.jpg)

<p align="center">Twisted Sponza rendered with panorama camera</p><br>

Toggle for this feature is compile-time. You can toggle the pragma `CAMERA_PANORAMA` in [`./src/common.h`](./src/common.h)

#### Post Processing

##### Gamma Correction

Implementing gamma correction is very trivial. But it is necessary if we want our final image to be correctly displayed on monitors, through which we see by our eyes.

##### Tone Mapping



### Performance

#### Fast Intersection: Stackless SAH-Based Bounding Volume Hierarchy

Ray-scene intersection is probably the best time consuming part of of ray tracing process. In a naive method we try to intersect every ray with every object in the scene, which is quite inefficient when there are numerous objects.

An alternative is to setup spatial acceleration data structure for the objects. 

I did two levels of optimization. These optimizations allow my path tracer to build BVH for the [Rungholt](https://casual-effects.com/data/) scene (6,704,264 triangles) in 7 seconds and run at 3 FPS, 2560 x 1440. (Click the links to see image details)

<table>
    <tr>
        <th><a href="./img/rungholt.jpg">Rungholt Rendering</a></th>
        <th><a href="./img/rungholt_bvh.png">Bounding Volume Hierarchy</a></th>
    </tr>
    <tr>
        <th><img src="./img/rungholt.jpg"/></th>
        <th><img src="./img/rungholt_bvh.png"/></th>
    </tr>
</table>

##### Better Tree Structure: Surface Area Heuristic 

First, I implemented a SAH-based BVH. SAH, the Surface Area Heuristic, is a method to determine how to split a set of bounding volumes into subsets when constructing a BVH, that the constructed tree's structure would be highly optimal.

##### Faster Tree Traversal on GPU: Multiple-Threaded BVH

The second level of optimization is done on GPU. BVH is a tree after all, so we still have to traverse through it during ray-scene intersection even on GPU.

#### Efficient Sampling: Sobol Quasi-Monte Carlo Sequence

In path tracing or any other Monte Carlo-based light transport algorithms, apart from improving

the performance from a point of view of programming, we can also improve it mathematically. Quasi-Monte Carlo sequence is a class of quasi-random sequence that is widely used in Monte Carlo simulation. This kind of sequence is mathematically proved to be more efficient than pseudorandom sequences (like what `thrust::default_random_engine` generates).

Theoretically, to maximize the benefit of Sobol sequence, we need to generate unique sequences for every pixel during each sampling iteration at real-time -- this is not trivial. Not to say that computing each number requires at most 32 bit loops. A better choice would be precomputing one pixel's sequence, then use some sort of perturbation to produce different sequences for different pixels.

Here is the result I get from testing the untextured [PBR texture scene](#representative-outcome). With the same number of samples per pixel, path tracing with Sobol sequence produces much lower variance (less noise).

<table>
    <tr>
        <th>Pseudorandom Sequence</th>
        <th>Xor-Scrambled Sobol Sequence</th>
    </tr>
    <tr>
        <th><img src="./img/sampler_indep.jpg"/></th>
        <th><img src="./img/sampler_sobol.jpg"/></th>
    </tr>
</table>

### Other

#### Single-Kernel Path Tracing

To figure out how much stream compaction can possibly improve a GPU path tracer's performance, we need a baseline to compare with. Instead of toggling streamed path tracer's kernel to disable stream compaction, we can separately write another kernel that does the entire ray tracing process. That is, we shoot rays, find intersection, shading surfaces and sampling new rays in one kernel.

#### First Ray Caching (G-Buffer)

In real-time rendering, a technique called deferred shading stores scene's geometry information in texture buffers (G-Buffer) at the beginning of render pass, so that . It turns out we can do something similar with offline rendering. 

## Performance Analysis

### Compare Streamed Path Tracing with Single-Kernel Path Tracing

What got me surprised it wasn't that efficient as expected. In some scenes, it was even worse than the single kernel path tracer. 

In general, it's a tradeoff between thread concurrency and time spent accessing global memory.



### Material Sorting: Why Slower

After implementing material sorting, I found it actually slower. And not by a little bit, but very significantly. With NSight Compute, I got to inspect how much time each kernel takes before and after enabling material sorting.

Like what the figure below shows, sorting materials does improve memory coalescing for intersection, sampling and stream compaction (I grouped sampling and lighting together because I did direct lighting). However, the effect is not sufficient to tradeoff the additional time introduced with sorting at all. As we can see the test result below, sorting makes up more than 1/3 of ray tracing time.

![](./img/sorted_no_sorted_camera.png)

Or, there is another possibility that BSDF sampling and evaluation is not that time consuming as expected. The bottleneck still lies in traversal of acceleration structure.

Therefore, in my opinion, material sorting is best applied when:

- There are many different materials in the scene
- Primitives sharing the same material are randomly distributed in many small clusters over the scene space. The clusters' sizes in solid angle are typically less than what a GPU warp can cover



### How Much GPU Improves Path Tracing Efficiency Compared to CPU

### Image Texture vs. Procedural Texture



## Third Party Credit

### Code

Apart from what was provided with the base code, two additional third party libraries are included:

- [*tinyobjloader*](https://github.com/tinyobjloader/tinyobjloader)
- [*tinygltf*](https://github.com/syoyo/tinygltf) (not used. I didn't implement GLTF mesh loading)

### Assets

[*Stanford Dragon*](http://graphics.stanford.edu/data/3Dscanrep/)

The following scene assets are licensed under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). They are modified to fit in the path tracer's scene file structure.

- [*Rungholt*](https://casual-effects.com/data/)
- [*Crytek Sponza*](https://www.cryengine.com/marketplace/product/crytek/sponza-sample-scene)

The rest of assets all have CC0 license from [*Poly Haven*](https://polyhaven.com/), a very good website where you can download free and public models, textures and HDR images.

## References





