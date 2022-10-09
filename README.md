CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Edward Zhang
  * https://www.linkedin.com/in/edwardjczhang/
  * https://zedward23.github.io/personal_Website/
 
* Tested on: Windows 10 Home, i7-11800H @ 2.3GHz, 16.0GB, NVIDIA GeForce RTX 3060 Laptop GPU

## Background
The Cuda Path Tracer Project - The Big Boi

## Representative Render
![](img/amogus.png)

### Requirements Fulfilled
#### Core Features
- Ideal Diffuse
- Specular (perfect and imperfect)
- Ray Termination using Stream Compaction
- Contiguous Memory Usage for Difference Materials
- Caching the first ray bounce
#### Additional Features
- OBJ Loading (with bounding box) (2)
- Refraction using Schlick approximation (2)
- Physically based Depth of Field effect (2)
- Antialiasing (2)
- Stratified Random Hemisphere Sampling (3)

## Ideal Diffuse


An ideal diffuse material shows the characteristics of being a perfectly smooth matte surface; approximated by lambertian shading in rasterization but it only judged by multiplicative color accumilation in this path tracer.
## Specular
Specular materials showcase the characteristics of being polished or shiny. Reflective in nature, meaning that rays that interact with this surface are reflected away from the surface in a more focused manner.

### Perfect
![](img/depthOfField.png)

Perfectly specular materials are like mirrors since reflection angles are deterministic.
### Imperfect
![](img/imperfSpec100ex[.png)

Specular Exponent: 100 (As the exponent increases, the surface gets closer and closer to being perfectly specular.

Imperfectly specular materials are just shiny since reflection angles are, while more focused than diffuse materials, nondeterministic.
## Stream Compaction
\[](img/RayTerminationChart.png)
Remove all rays that hit nothing in the scene.
## Material Sorting in Memory
\[](img/MaterialSortChart.png)

Sort all ray intersections by the materials they hit with the intent of enabling memory coalescence for bsdf interactions based on different materials.
## Ray Caching
\[](img/RayCacheChart.png)

The first rays cast into the scene are deterministic based on the camera's position; thus there is no need to recalculate their interactions and intersections for each iteration that the Pathtracer renders.
## OBJ Loading
\[](img/basicObjTest.png) 
Tested this implementation against importing a basic cube that was stored as an OBJ.

Implemented functionaltiy loads in an arbitrary, but untextured mesh. Implemented bounding box acceleration data structure as well.

\[](img/ObjPerformance.png)
Performance is significantly improved when loading the Low Poly Among Us Crewmate obj into the scene as shown in chart above

## Refractive
![](img/Transmissive.png)

Refractive materials like glass both refract and reflect light. Using schlick approximation, we nondeterministically have some rays get refracted into the object while other rays are reflected away.

## Depth of Field
![](img/Screenshot 2022-10-08 202552.png)
Focal Dist of 12
Lens Radius of 1

Lenses in real life have radii and focal distances (based on their concavity) which bend light rays in a way that only certain objects a certain distance away are actually in focus. This is approximated using random disk sampling to artificially bend cameras cast into screen space.

## Antialiasing

Sending out multiple rays approximated by small scale directional jitter per camera ray per pixel to get sub-pixel values.

![](img/AAClose.png)
Close up of a sphere with Anti Aliasing applied.
![](img/withoutAAClose.png)
Close up of a sphere without Anti Aliasing applied.

The difference is subtle but the curved surface is a little less jagged with anti-aliasing applied.

## Stratified Sampling
Improve random hemisphere sampling by using a grid to stratify the portions of the hemisphere upon which we sample ray directions.
\[](img/StratSampling.png)

Random ray bounces employing stratified random hemisphere sampling.

\[](img/RandomSampling.png)

Standard random ray bounces used for random hemisphere sampling.

The shadows are slightly different in terms of noise levels, though they converge to very similar results.


## References

* [PBRT] Physically Based Rendering, Second Edition: From Theory To Implementation. Pharr, Matt and Humphreys, Greg. 2010.
* Antialiasing and Raytracing. Chris Cooksey and Paul Bourke, http://paulbourke.net/miscellaneous/aliasing/
* [Sampling notes](http://graphics.ucsd.edu/courses/cse168_s14/) from Steve Rotenberg and Matteo Mannino, University of California, San Diego, CSE168: Rendering Algorithms

## Bloopers
![](img/GlassBlooper.png)
![](img/GlassBlooper1.png)
