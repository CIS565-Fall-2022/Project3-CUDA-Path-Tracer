CUDA Path Tracer
================

(leading img here)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Evan S
* Tested on: Strix G15: Windows 10, Ryzen 7 4800H @ 2.9 GHz, GTX 3050 (Laptop)

## Background
This program takes in a text file describing a scene and renders it with the canonical path tracing algorithm. It supports pure specular and pure diffuse materials.   

Other supported toggle-ables include:


For each extra feature, you must provide the following analysis:

Overview write-up of the feature along with before/after images.
Performance impact of the feature
If you did something to accelerate the feature, what did you do and why?
Compare your GPU version of the feature to a HYPOTHETICAL CPU version (you don't have to implement it!)? Does it benefit or suffer from being implemented on the GPU?
How might this feature be optimized beyond your current implementation?





* Stream compaction
* Sorting rays by intersected material type
* Caching first intersection

## Performance Analysis

### Refractions

Material types can be specified in the scene text files. Dielectric describes a nonconducting material(e.g., glass). Here, a type of DIELECTRIC simply means it both reflects and refracts light according to a specified "index of refraction". For simpler calculations, I used the [Schlick approximation](https://en.wikipedia.org/wiki/Schlick%27s_approximation), of the Fresnel equations since it achieves results at up to 32x speed for less than 1% average error (further reading: References - Ray Tracing Gems II).    
This refractive BSDF is an inherent part of the ray tracing evaluation; any CPU path tracer that tries to implement the same thing will be orders of magnitude slower.  
Theoretically, adding other approximations could be faster, at the cost of physical accuracy.

The speed/FPS of this material is comparable to diffuse or mirrored. 
| Material | Diffuse | Mirrored | Dielectric | 
| :------- | :-------: | :-------: | :-------: |
| Frames per second | 51.8 | 52.6 | 52.1 |
| Scene | <img src="img/diffuse_bench.png"> | <img src="img/mirror_bench.png"> | <img src="img/dielectric_bench.png"> |

Note the subtle reflection of the light on the dielectric spheres; that is the light reflection contribution.   

### Core Feature Benchmarks

A basic Cornell box scene, used for benchmarks.     
<img src="img/basic_cornell_box.png" width=30% height=30%>

#### Stream compaction
Stream compaction/partitioning of terminated paths allows warps to go offline early. The below graph gives the average FPS (a benchmark of the speed of the path tracing) of path tracing with and without stream compaction as depth of the tracing goes up on the (above) basic Cornell box scene.   
<img src="img/compacted_vs_not.png" width=70% height=70%>    
The chart indicates that while tracing depth is low, the overhead of partitioning the rays presents a huge slowdown. However, when depth is higher (when it is expected that rays have to bounce numerous times before terminating), the ability to free up threads operating on terminated rays from stream compaction becomes worthwhile. 

#### Sorting rays by intersected material type
Sorting rays by the material type they intersect with means that the code path of contiguous threads (warps) will be grouped together when calculating the BSDF, ensuring parallelization and therefore performance improvement. The average FPS of this sorting on the basic Cornell box scene as well as a modified box scene(see scenes/cornell_more_objs_for_sort_bench.txt) with many spheres, some diffusive and some reflective, is noted in the table below: 
| (FPS) | Sorting | No Sorting |
| :------- | :-------: | :-------: |
| Basic scene | 9.1 | 75.3 |
| More materials | 9 | 70.4 |

Clearly sorting causes enormous slowdown on these simple scenes. The slight dip in the more materials with sorting compared to more materials with no sorting implies that the two different objects causes some serialization in the program, where thread paths have to go down either the diffusive or specular BSDF. With a greater variety of materials and objects causing further serialization, sorting can be a worthwhile improvement. 

#### Caching first intersection
The first intersection of rays is static, as it is initiated by the static ray casts per-pixel of the render. Caching the first intersection for use across subsequent iterations is expected to boost performance overall at a slight memory cost. The table below compares the average time(over 5 tries) needed for 5000 full iterataions between caching and no caching for the basic Cornell box scene:
| (seconds) | Caching | No caching |
| :------- | :-------: | :-------: |
| Basic scene | 69 | 70 |   

Curiously, the difference is negligible, though weighed toward the caching. The simplicity of the Cornell box scene makes it difficult to ascertain clear winners when it comes to different rendering options.

## References
[Ray Tracing Gems II](http://www.realtimerendering.com/raytracinggems/rtg2/)
[Ray Tracing in One Weekend](https://raytracing.github.io/)     
[Fresnel Equations, Schlick Approximation, Metals, and Dielectrics](http://psgraphics.blogspot.com/2020/03/fresnel-equations-schlick-approximation.html)