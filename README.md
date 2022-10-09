CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Wenqing Wang
  * [LinkedIn](https://www.linkedin.com/in/wenqingwang0910/) 
* Tested on: Windows 11, i7-11370H @ 3.30GHz 16.0 GB, GTX 3050 Ti

## Overview
This project implemented a GPU-based path tracer using CUDA with several visual and performances improvements.

![main_with_des](https://user-images.githubusercontent.com/33616958/194778938-be4f9d29-40d6-491c-b5fc-df4efbe4aff5.jpg)


#### Feature Implemented:
* Visual
    * Shading kernel with BSDF evaluation for diffuse, specular-reflective and refractive surfaces.
    * Stochastic Sampled Antialiasing
    * Depth of Field
    * Motion Blur
* Mesh
    * Arbitrary mesh loading and rendering based on tinyOBJ with toggleable bounding volume intersection culling
* Performance
    * Path continuation/termination using streaming compaction
    * Material sorting
    * First bounce caching
    
## Features
### Shading Kernel with BSDF Evaluation (diffuse, reflect & refract)
In the figure below, the cube is given a diffuse material, and the sphere is given a refractive material with IOR = 1.5. The floor is set to be fully reflective.

<img alt="img1" src="https://user-images.githubusercontent.com/33616958/194779389-f31b28f2-9e31-4af2-bba2-555c74d73a02.png">

### Stochastic Sampled Antialiasing

|Without Antialiasing| With Antialiasing|
|--|--|
|![image](https://user-images.githubusercontent.com/33616958/194782019-3a05d9fc-2903-400c-877a-7313a8ec13d5.png)|![image](https://user-images.githubusercontent.com/33616958/194781983-c60c21e4-e625-4c5c-8517-9574722ced32.png)


### Depth of Field
|Focal Distance = 8.0 |Focal Distance = 12.0 |
|--|--|
|<img width="356" alt="dof_0 4_8" src="https://user-images.githubusercontent.com/33616958/194781025-26f6b5c4-dc5f-4533-ab4e-891724d83587.png"> |<img width="353" alt="dof_0 4_13" src="https://user-images.githubusercontent.com/33616958/194781014-679f34e0-d70e-4914-bdfb-505f828505ff.png"> |


### Motion Blur
|Without Motion Blur| Motion Blur|
|--|--|
|<img width="356" alt="nmb" src="https://user-images.githubusercontent.com/33616958/194783108-f1cd9ee8-7c09-41c8-96a4-cde9d3e6f4bb.png"> | <img width="356" alt="nmb" src="https://user-images.githubusercontent.com/33616958/194783120-dbb1457f-21a1-441e-b3ad-9f039eba613e.png">  |



