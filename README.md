CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Haoquan Liang
  * [LinkedIn](https://www.linkedin.com/in/leohaoquanliang/)
* Tested on: Windows 10, Ryzen 7 5800X 8 Core 3.80 GHz, NVIDIA GeForce RTX 3080 Ti 12 GB

![Overview](img/overview.png)


# Table of Contents  
[Features](#features)   
[Feature Showcase](#showcase)
[Performance Analysis](#perf_anal)  

# Features
<a name="features"/>   

## Core features
* Ideal Diffuse surfaces
* Perfectly specular-reflective
* Imperfect specular-reflective
* Path termination with Stream Compaction
* Sorting rays by materials
* Caching the first bounce intersections
## Additional features
* Refraction
* Physically-based depth-of-field
* Stochastic Sampled Antialiasing
* Texture mapping with a basic procedural texture
* Arbitrary obj mesh loading
* Direct lighting
* Motion blur
* Post-processing shaders (greyscale, sepia, inverted, and high-contrast filters)

# Feature Showcase
<a name="showcase"/>

## BSDF Evaluation for Ideal Diffuse Surface
Ideal Diffuse   
![ideal diffuse](img/diffuse.png)
Perfectly Specular Reflective   
![perfectly specular](img/perf-specular.png)
Imperfect Specular Reflective   
![imperfect specular](img/imperf-specular.png)
## Path Termination using Stream Compaction
## Sorting Rays
## Caching the First Bounce Intersections
## Mesh Loading
![mesh](img/mesh.png)
## Refraction
![Refraction](img/refraction.png)
## Depth of Field
![DOF](img/dof.png)
## Stochastic Sampled Anti-aliasing
![Anti-aliasing](img/antialiasing.png)
## Texture Mapping with Simple Procedural Texture
 ![Texture-mapping](img/texture.png)
## Direct Lighting
![Directlighting](img/directlighting.png)
## Motion Blur
![motionblur](img/motionblur.png)
## Final Rays Post-processing
Greyscale   
![Greyscale](img/greyscale.png)
Sepia   
![Sepia](img/sepia.png)
Inverted   
![Inverted](img/inverted.png)
High-contrast   
![High-contrast](img/high-contrast.png)
