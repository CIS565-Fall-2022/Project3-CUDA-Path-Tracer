CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Guanlin Huang
  * [LinkedIn](https://www.linkedin.com/in/guanlin-huang-4406668502/), [personal website](virulentkid.github.io/personal_web/index.html)
* Tested on: Windows 11, i9-10900K @ 4.9GHz 32GB, RTX3080 10GB; Compute Capability: 8.6

## Representive Scene
![](img/title.png)

- Scene file: title.txt
- Resolution : 1024 x 1024
- Iteration : 5000

* GLTF mesh loading
* Texture mapping
* Physically based depth of field

## Features
- Ideal Diffuse
- Perfectly specular-reflective
- Physically-based depth-of-field 
- Antialiasing
- Texture and Normal mapping
- Arbitrary mesh loading and rendering
- Path continuation/termination with stream compaction
- contiguous in memory by material type
- cache the first bounce intersections

## Feature Descriptions
### Ideal Diffuse
The light will bounce off a diffuse object in random directions, evenly dispersed in a hemisphere around the intersection point. The result is a surface that looks like plastic.
![](img/dif.png)

### Perfect Reflective