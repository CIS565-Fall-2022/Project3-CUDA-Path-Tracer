CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**


* Yilin Liu
  * [LinkedIn](https://www.linkedin.com/in/yilin-liu-9538ba1a5/)
  * [Personal website](https://www.yilin.games)
* Tested on personal laptop:
  - Windows 10, Intel(R) Core(TM), i7-10750H CPU @ 2.60GHz 2.59 GHz, RTX 2070 Max-Q 8GB

Features
=============
* Shading Kernel with BSDF Evaluation
* Path Termination using Stream Compaction
* Sorting of Paths Segment by Material type
* First Bounce Intersection Cache
* Uniform diffuse
* Perfect Specular Reflective
* Refraction (Fresnel dielectric)
* Stochastic Sampled Antialiasing
* OBJ Mesh Loading using TinyOBJ
* Physically-based depth-of-field
* Motion Blur

Problems of Performing BSDF in a big kernel 
============
CUDA can only launch a finite number of blocks at a time. Some threads end with only a few bounces while others may end with a lot. Therefore, we will waste a lot of threads. 

To solve this problem, we  launch a kernel that traces ONE bounce for every ray in the pool. According to the results, we remove terminated rays from the ray pool with stream compaction. 

Sort by Materials Type 
============
Using Radix Sort by material ID, we can batch rays according to material type. Therefore, we can further parallelize rays and perform intersection testing and shading evaluation in separate kernels. 

First Bounce Intersection Cache
============
We further cache the first bounce intersection and store it in a buffer. Later bounces can use it since this bounce stays the same regardless of iterations. 

Refraction
===========
The refraction effects was implemented using glm's `refract()` function according to Schlick's approximation and Snell's Law. 

| Refraction Ball | Refraction Glass Bottle |
:-------:|:-------:
|![](img/refraction.png)|![](img/refraction2.png)|

Anti-aliasing
===========
I jitter the direction of sample ray to reduce the aliasing effects

| With AA | No AA |
:-------:|:-------:
|![](img/AA.png)|![](img/NO_AA.png)|
|![](img/aa_large.png ){:height="800px" width="800px"} | ![](img/noAA_large.png){:height="800px" width="800px"}|

Mesh Loading
===========
I used [tinyObj](https://github.com/tinyobjloader/tinyobjloader) to load obj file.

| Bunny | No AA |
:-------:|:-------:
|![](img/AA.png)|![](img/bunny_refract.png)|

Depth of Field
============

Motion Blur
===========
I developed two ways to achieve motion blur in path tracer. The first one works on the camera and it is global based.

| Sphere Motion Blur 1 | Sphere Motion Blur 2 |
:-------:|:-------:
|![](img/motion_blur_object.png)|![](img/motion_blur_object2.png)|


The second one works on certain objects and users can define a direction of the movement. 

| Camera Motion Blur 1 | Camera Motion Blur 2 |
:-------:|:-------:
|![](img/motion_blur_camera.png)|![](img/motion_blur_camera2.png)|

Analysis
===============

Snapshots
===============


Bloopers
===============
  |![image](img/bloopers/refract_fail.png)|
  |:--:| 
  | *Refraction Fail* |
  
  |![image](img/bloopers/cow1.png)|
  |:--:| 
  | *Lonely Cow (Bounding Box Predicate Fail)* |