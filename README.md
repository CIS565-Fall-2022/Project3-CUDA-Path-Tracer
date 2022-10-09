CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Hanlin Sun
* Tested on: Windows 10, i7-8750H @ 2.22GHz 1024GB, NVIDIA Quadro P3200

## Background

This is a GPU Path Tracer implemented in C++/CUDA, based on fundamental PBR raytracing knowledge.
It allows loading GLTF files for meshes.

## Screen Shots

![Render Img](img/Final.JPG)

## Basic Features

### Basic Shading

First we need to convert the naive path tracing shading technique from CPU version to GPU version.This allows for global illumination
effect to be achieved. (But very slow)

Two basic shadings were implemented first: diffuse and perfect specular. The diffuse component is shaded by generating new ray using cosine weighted random direction in a hemisphere, and the perfect specular uses glm::reflect to generate a perfectly reflected ray.

Diffuse | Specular
:--------------------------:|:------------------------:
![Render Img](img/Diffuse.JPG) | ![Render Img](img/specular.JPG)

### Refraction

To implement refraction, we need to use snell's law and fresnel law to compute material's eta, then use fresnel formula to compute final ray color.

Pure Specular | Specular with Refraction
:--------------------------:|:-------------------------:
![Render Img](img/specular.JPG) | ![Render Img](img/refraction.JPG)

### SSAA & Depth of Field

These methods is used to increase visualize experience. Depth of Field seems a little bit buggy,but SSAA works pretty well and it looks good.

SSAA not Enabled | SSAA Enabled
:-------------------------:|:-------------------------:
![Render Img](img/enableSSAA.JPG) | ![Render Img](img/NOSSAA.JPG)

DoF not Enabled | Dof Enabled
:-------------------------:|:-------------------------:
![Render Img](img/NOSSAA.JPG) | ![Render Img](img/DoF.JPG)

### String Compaction

Since we are using naive path tracing method, many rays may reflect out of the scene and become useless computation.
To optimize it, we need to remove these useless ray from path tracer.

In this project I use the `thrust::stable_partition` kernal to partition these paths based on completion.

This method is really useful for unclosed scene and bring performance improvement to the MCPT since the GPU will compute less rays after each bounce.



But for closed scene, since no ray will bounce outside the scene, this method will not bring performance improvement in this case.

### Material Sorting



### First bounce Caching



### GLTF Mesh Loading




## Reference
