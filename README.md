CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Chang Liu
  * [LinkedIn](https://www.linkedin.com/in/chang-liu-0451a6208/)
  * [Personal website](https://hummawhite.github.io/)
* Tested on personal laptop:
  - i7-12700 @ 4.90GHz with 16GB RAM
  - RTX 3070 Ti Laptop 8GB

## Introduction

This is our third project of CIS 565 Fall 2022. In this project, our goal is to implement a GPU-accelerated ray tracer with CUDA. 

## Representative Outcome

![](./img/photo_realistic.jpg)

<p align="center">"Photoly" realistic!</p>

| Scene Specs             | [[./scenes/pbr_texture.txt]](./scenes/pbr_texture.txt) |
| ----------------------- | ------------------------------------------------------ |
| Resolution              | 2400 x 1800                                            |
| Samples Per Pixel       | 3000                                                   |
| Render Time             | < 7 minutes (> 7.5 frames per second)                  |
| Million Rays Per Second | > 32.4                                                 |
| Triangle Count          | 25637                                                  |
| Lighting                | HDR Environment Map                                    |



![](./img/aperture_custom.jpg)

<p align="center">Star-shaped bokehs</p>



## Features

### Visual

#### Direct Lighting with Multiple Importance Sampling

#### Importance Sampled Skybox (Environment Map)

Tired of "virtual artificial" light sources? Let's introduce some real-world li

#### Physically-Based Materials

#### Normal Map & PBR Texture

#### Physically-Based Camera: Depth of Field & Custom Bokeh Shape

This is my favorite part of the project.

| No DOF                      | DOF                     |
| --------------------------- | ----------------------- |
| ![](./img/aperture_off.jpg) | ![](./img/aperture.jpg) |

This idea can even be extended by stochastically sampling a masked aperture image instead of the whole aperture disk. 

<div align="center">
	<img src="./scenes/texture/star3.jpg" width="15%"/>
	<img src="./scenes/texture/heart2.jpg" width="15%"/>
</div>

| Star Mask                      | Heart Mask                    |
| ------------------------------ | ----------------------------- |
| ![](./img/aperture_custom.jpg) | ![](./img/aperture_heart.jpg) |

#### Efficiently Sampling: Sobol Low Discrepancy Sequence

| Pseudorandom Sequence        | Xor-Scrambled Sobol Sequence |
| ---------------------------- | ---------------------------- |
| ![](./img/sampler_indep.jpg) | ![](./img/sampler_sobol.jpg) |



#### Post Processing

##### Gamma Correction

##### Tone Mapping

---



### Performance

#### Fast Intersection: Stackless SAH-Constructed BVH

For ray-scene intersection, I did two levels of optimization.

First, I wrote a SAH-based BVH. SAH, the Surface Area Heuristic is a method to decide how to split a set of bounding volumes 

The second level of optimization

#### Single-Kernel Path Tracing

There is a paper . It had an interesting opinion: instead of 

### Other

#### Streamed Path Tracing Using Stream Compaction

#### First Ray Caching (G-Buffer)

Since I implemented anti-aliasing and physically based camera at the very beginning, when I noticed that there is still a requirement in the basic part, I found it 

## Performance Analysis

### How Much GPU Improves Path Tracing Efficiency

I'm able and confident to answer this question because I have one CPU path tracer from undergrad. 

### Why My Multi-Kernel Streamed Path Tracer Not Faster Than Single-Kernel?

To know how streaming the rays can improve path tracing efficiency, I additionally implemented a single-kernel version of this path tracer.

What got me surprised it wasn't efficient as expected. In some scenes, it was even worse.

Using NSight Compute, I inspected 

In general, it's a tradeoff between thread concurrency and time spent accessing global memory.

### Material Sorting: Why Slower

Or, there is another possibility that BSDF sampling and evaluation is not that time consuming as expected. The bottleneck still lies in traversal of acceleration structure.

Therefore, in my opinion, material sorting is best applied when:

- There are many different materials in the scene
- Primitives sharing the same material are randomly distributed in many small clusters over the scene space. The clusters' sizes in solid angle are typically less than what a GPU warp can cover

### Image Texture vs. Procedural Texture
