CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Di Lu
  * [LinkedIn](https://www.linkedin.com/in/di-lu-0503251a2/)
  * [personal website](https://www.dluisnothere.com/)
* Tested on: Windows 11, i7-12700H @ 2.30GHz 32GB, NVIDIA GeForce RTX 3050 Ti

![](img/main.png)

## Introduction

In this project, I implemented a CUDA path tracer for the GPU. Previously in Advanced Rendering, I implemented a Monte Carlo Path Tracer for the CPU. In this project, the path tracer is

![](img/mainAntialiasing.png)

## Core Features
1. Shading kernel with BSDF Evaluation for Diffuse and Perfect/Imperfect Specular
2. Path continuation/termination using Stream Compaction
3. Contiguous arrangement of materials based on materialId
4. First bounce caching

![](img/part1Final.png)
![](img/part1FinalSpecular.png)
![](img/imperfectSpecular.png)

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

## Bloopers! :)

Too much antialiasing: me without my contacts

![](img/antialiasing1.png)
