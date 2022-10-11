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

### Stream Compaction Ray Termination

### First Bounce Caching

### Depth of Field

![](img/renders/depth_of_field.PNG)

### Stochastic Anti-Aliasing

![](img/figures/no_aa.PNG)

![](img/figures/aa.PNG)

### Material Sorting

One optimization that 

### Russian Roulette Ray Termination

### Tone Mapping and Gamma Correction

![](img/renders/no_hdr_no_gamma.PNG)

![](img/renders/hdr_no_gamma.PNG)

![](img/renders/no_hdr_gamma.PNG)

![](img/renders/diffuse.PNG)

### OBJ Loading with TinyOBJ

![](img/renders/wireframe.PNG)

![](img/renders/diffuse.PNG)

### Bounding Volume Hierarchy (BVH)

## Performance Analysis

## Bloopers

![](img/bloops/lotsadragons.PNG)

![](img/bloops/bloopmat2.PNG)

![](img/bloops/fresnel.PNG)

![](img/bloops/refractiondoesntwork.PNG)

