CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Shixuan Fang  
  - [LinkedIn](https://www.linkedin.com/in/shixuan-fang-4aba78222/)
* Tested on: Windows 11, i7-12700kf, RTX3080Ti (Personal)

## Project Overview
<p align="center">
  <img src="/img/final_picture.png" width="400" height="400">
</p>

In this project, I implemented a CUDA-based path tracer, which is very different from CPU path tracer. Pathtracing in CPU is used to be recursive, but since CUDA doesn't support recursion(or very slow on new GPUs), this path tracer uses a iterative way.
<p align="center">
  <img style="float: right;" src="https://user-images.githubusercontent.com/54868517/194418589-a882f9be-abda-4ae0-afe7-5b39bb03791e.png" width="500" height="200">
</p>

Also, traditional CPU pathtracing is down pixel by pixel, but as the figure shows, doing the same algorithm(parallel by pixel) on GPU will cause different threads finish in different time and therefore cause many divergence. The solution, then, is to parallel by rays. We launch a kernel that traces **one** bounce for every ray from a ray pool, update the result, and then terminate rays with stream compaction


## Core Features

### 1. Basic shading

There are three basic shadings implemented --- pure diffuse, pure reflection, and both reflection and refraction. Diffuse is done by generating a new direction randomly across the hemisphere, reflection is done by reflecting the incident ray direction by the surface normal, and refraction is done by using Schlick's approximation to create Fresnel effect.
<p align="center">
  <img src="https://user-images.githubusercontent.com/54868517/194421652-410c82f6-60c2-4ca4-8200-2039355fc622.png" width="300" height="300">
</p>

### 2. Acceleration

- Stream Compaction

The first acceleration is to remove unnecessary rays that are terminated(remaining bounce = 0) by using Stream Compaction. In this project I used ```thrust::stable_partition``` to compact.

- Sort rays by material type

The second acceleration is to sort all rays by their material types, which can result in good memory access pattern for CUDA. In this project I used ```thrust::stable_sort_by_key``` to sort.

- Cache the first bounce for re-use

The third acceleration is to cache the first bounce, therefore all other iterations can use the same data and save some performance.


### 3. Mesh loading with tinyObj


### 4. Microfacet shading model


### 5. Stochastic Sampled Antialiasing


### 6. Physically-based Depth-of-field

### 7. Direct Lighting
