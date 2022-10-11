CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Nick Moon
  * [LinkedIn](https://www.linkedin.com/in/nick-moon1/), [personal website](https://nicholasmoon.github.io/)
* Tested on: Windows 10, AMD Ryzen 9 5900HS @ 3.0GHz 32GB, NVIDIA RTX 3060 Laptop 6GB (Personal Laptop)


**This is an offline Physically-Based Path Tracer accelerated on the GPU with CUDA. Path Tracing is a
technique of simulating the actual way light bounces around a scene from light sources towards an eye or
camera. Physically-Based implies energy conservation and plausibly accurate light simulation via solving
the Light Transport Equation originally proposed by Kajiya.** \
\
**I used this project to: learn path tracing
optimization techniques by utilizing the parallel nature of the GPU; review, reimplement, and extend
my work from CIS 561: Physically-Based Rendering; use this as an opportunity to learn and implement the
Bounding Volume Hierarchy acceleration structure, keeping in mind the limitations of GPU programming.**

**The result of the Multiple Importance Sampling (MIS) and Bounding Volume Hierarchy (BVH) I implemented is a PBR path tracer capable
of rendering scenes with millions of triangles and multiple area light sources with at least one sample
per second.**

## RESULTS

![](img/renders/objs.PNG)

![](img/renders/teapot.PNG)

![](img/renders/dragons.PNG)

## IMPLEMENTATION

### Physically-Based Rendering

The following explanation is borrowed from Adam Mally's explanation of the Light Transport Equation

![](img/figures/adams_LTE.PNG)

### Bidirectional Scattering Distribution Functions (BSDFs)

Bidirectional Scattering Distribution Functions describe the scattering of energy on a surface as a result
of the surface's material properties. Components of a BSDF can include the 
Bidirectional Reflectance Distribution Function and the Bidirectional Transmittance Distribution Function.
These describe the reflective and transmissive properties respectively.

#### Diffuse BRDF

The Diffuse BRDF scatters light in all directions

![](img/renders/diffuse.PNG)

#### Specular BRDF

![](img/renders/spec_brdf.PNG)

#### Specular BTDF

![](img/renders/spec_btdf.PNG)

#### Specular Glass BxDF

![](img/renders/spec_glass.PNG)

#### Specular Plastic BxDF

![](img/renders/spec_plastic.PNG)

### Multiple Importance Sampling (MIS)

![](img/renders/depth_1.PNG)
![](img/renders/depth_2.PNG)
![](img/renders/depth_3.PNG)
![](img/renders/depth_4.PNG)
![](img/renders/depth_5.PNG)
![](img/renders/depth_8.PNG)

![](img/renders/iter_1.PNG)
![](img/renders/iter_5.PNG)
![](img/renders/iter_10.PNG)
![](img/renders/iter_50.PNG)
![](img/renders/iter_100.PNG)
![](img/renders/iter_100_mis.PNG)

### Depth of Field

![](img/renders/depth_of_field.PNG)

### Stochastic Anti-Aliasing

![](img/figures/no_aa.png)

![](img/figures/aa.png)

### Tone Mapping and Gamma Correction

The image that results from rendering this scene with a basic RGB path tracer is the following image:

![](img/renders/no_hdr_no_gamma.PNG)

This image looks overly dark in the areas further from the light, and is too bright near the top of the 
teardrop mesh. It is hard to tell there is even any global illumination going on! This is because
this render has not been tone mapped nor gamma corrected.

This is the result of adding Reinhard HDR tone mapping to the render:

![](img/renders/hdr_no_gamma.PNG)

The intense color near the tip of the teardrop has been eased out, providing a much more natural looking
diffuse gradient from top to bottom.


This is the result of adding gamma correction to the render:

![](img/renders/no_hdr_gamma.PNG)

The color near the edge of the screen has now been boosted to more closely match human color perception,
and now those areas are now all clearly illuminated by a mix of global illumination and the far away light source.
However, without the Reinhard operator, the tip of the teardrop is still to bright.

Finally, combining both operations together yields:

![](img/renders/diffuse.PNG)

This render has been properly tone mapped and gamma corrected, and now looks more cohesive, natural, and
follows physically-based rendering principles.

### OBJ Loading with TinyOBJ

![](img/renders/wireframe.PNG)

P.S. In order to render this wireframe version while still in a physically-based framework and in an
easy and quick to implement way, I changed my triangle intersection test. In the final check for if
the barycentric coordinates of the hit point are within the triangle, I simply added an upper limit to
the distance from an edge a point is that will still count as an intersection.



![](img/renders/diffuse.PNG)


### Optimizations Features
 
#### Bounding Volume Hierarchy (BVH)

During path tracing, a ray needs to intersect with the geometry in the scene to determine things like
the surface normal, the distance from the origin, and the BSDF at the surface. To do this, a naive
ray tracing engine has to perform a ray-primitive intersection test for every primitive in the scene.
While this is acceptable for scenes with a small amount of, for example, cubes and spheres, this quickly falls
apart for scenes with meshes, especially ones made up of thousands and more triangles.

The Bounding Volume Hierarchy is a binary-tree based primitive acceleration structure that can be used
to optimize this ray-scene intersection test. Instead of intersecting with every triangle, instead a ray
interesects with nodes in the BVH, represented spatially as an Axis Aligned Bounding Box (AABB). An intersection
test will only be performed on a node if the intersection test with its parent was found to be a hit. The
leaf nodes store the actual primitives, so the triangle intersection test, which is more expensive than
the AABB

#### Russian Roulette Ray Termination

Russian Roulette Ray Termination is an optimization that seeks to remove ray paths that have no additional
information to contribute to the scene by bouncing more times. For my renderer, I enable a ray termination
check if the depth is greater than 3. This is because, for a scene made of mostly diffuse surfaces,
4-5 bounces is usually enough to get the majority of the global illumination information for a path (assuming
MIS/direct light sampling is also used). Thus, on depth 4 and greater, a random number is generated for each
ray path. If the max channel of the throughput of the ray (the value which gets attenuated when hitting
diffuse surfaces) is less than this random number, then the ray is terminated. However, if the max channel
is greater, then the ray path continues bouncing (for at least one more bounce if applicable). Additionally,
the throughput of a ray path that passes this check is divided by it max channel. This is to counter
the "early" termination of the rays which did not pass this check on earlier (or later) samples, thus
preserving the overall light intensity at each pixel. The image would be slightly darkened without this,
as energy would no longer be conserved.

#### Stream Compaction Ray Termination

The following explanation is from my HW 02: Stream Compaction README:

"Stream compaction is an array algorithm that, given an input array of ints ```idata``` of size ```n```,
returns an output array ```odata``` of some size ```[0,n]``` such that ```odata``` contains only the
values ```x``` in ```idata``` that satisfy some criteria function ```f(x)```. This is essentially 
used to compact an array into a smaller size by getting rid of unneeded elements as determined by 
the criteria function ```f(x)```. Values for which ```f(x)``` return ```true``` are kept, while
values for which ```f(x)``` return false are removed."

For the purposes of path tracing, we have an array of ray paths, with the remaining bounces of a path
being the amount of bounces a ray has left to take in the scene. We obviously don't want to be doing
unecessary intersections and shading for rays that no longer contribute color to the final
image. We thus define the ```f(x)``` from Stream Compaction to be that a ray path's remaining bounces
is not equal to 0. By using the Thrust library's Stream Compaction function, we can thus move unneeded rays
to the back of the array of paths, and only call the intersection and shading kernels on the rays that
will actually contribute to the rendering.

#### Material Sorting

Another optimization that can be made is by recognizing that all the material shading is currently done in
a single "uber" kernel. This means that individual threads in a warp might be calculating the f, pdf, and wi
terms for the hit surface using different BSDFs. These BSDFs could potentially be very intsense and vary with
high spatial frequency across the scene. This means a lot of potential warp divergence, which could dramatically
slow down the shading kernel. Instead, if we sort the ray path's and intersections by the material type
returned by the intersection kernel, we could then insure that most ray paths with similar material
types are laid out sequentially in memory. This will yield less divergence, and thus faster runtime. Thrust
Radix Sort is used for this sorting process. Note that the material IDs are what are being sorted here,
not necessarily the BSDF or material type, so unfortunately materials that are the same in every aspect save
albedo will still count as seperate entries to be sorted.

#### First Bounce Caching

First bounce caching is an optimization used in settings where features like depth of field or anti-aliasing
are not needed (which I assume are pretty rare.) Because the direction and origin of the intitial camera
ray are deterministic in this scenario, then this means this ray will always hit the same primitive. Thus,
instead of computing this intersection every sample, we instead cache this first intersection into a seperate
device array on the first sample, and then load it into the main intersection device array for every sample
beyond the first. Especially for scenarios where the max ray depth is on the lower end, this should
improve the runtime.

## Performance Analysis

## Bloopers

![](img/bloops/lotsadragons.PNG)
Accidentally made the walls specular

**For more bloopers, see img/bloops :)**

