CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Dongying Liu
* [LinkedIn](https://www.linkedin.com/in/dongying-liu/), [personal website](https://vivienliu1998.wixsite.com/portfolio)
* Tested on:  Windows 11, i7-11700 @ 2.50GHz, NVIDIA GeForce RTX 3060

# Project Description
In this project. I implemented a simple GPU path tracer with CUDA. For each iteration, instead of evaluating every ray all in one for loop on CPU, I paralleled it by rays. 

Here's what will be done in one iteration:

**1) Generate rays from camera into the scene**

**2) For each ray, compute the intersection with scene** 

**3) For each ray, shade and bounce the ray base on the material BSDF until the ray's bouncing limit is reached or hits a light**

The iteration will stop when max depth is reached or all the rays are terminated.

Having implemented a path tracer on CPU before, it's nice to see how things go parallel on GPU and how fast it could be when running on GPU. I will elaborate the features I implemented in the follwing paragraphs.

# Features
## Diffuse, Reflection and Refraction Surfaces
In my project, I implemented three types of materials, ideal diffuse, perfectly specular-reflective and refractive.

For the ideal diffuse surface, when scattering rays, the direction is radomnly picked using the cosine-weighted scatter function.

For perfectly specular-reflective surfaces, when scattering rays, there is only one direction which is the reflected ray and it is calculated using the glm::reflect function.

For refractive surfaces, the ray scattering is more complecated. When scattering the rays, I used the Schlick's approximation to simply ditermine whether use the ray refract direction or reflect direction.

## Anti-aliasing
We can do anti-aliasing with super sampling, and in path tracer, super sampling can be implemented with only a slightly change of code.

For one pixel, the final color is the average color of every iteration's result of that pixel. In the former implementation, at the beginning of every iteration, all the ray are shooting into the scene along the same direction, so the average result is all for on point on that pixel square. 

To implement anti-aliasing, when generating ray from camera, I slightly noised the ray direction so that the results of every iteration are for different points on that pixel square. By doing this, we can get the super sampling anti-aliasing easily.

## Motion Blur
I implemented motion blur in a simple way. When calculating the intersection of ray, I randomly noised the ray origin along a vector, which I defined as MOTION_VELOCITY, so the rendering result looks like the object is moving. 

## Depth of Field
Instead of casting rays from a single point on the camera, we considered the camera as a small lens. We cast rays from various points on the lens. When an object is further away or closer to the focal plane, the ray that cast coresponding to a single pixel hit the object in a large vairaty of locations, we then average all the colors into a single pixel. This will result in a blurry image. 

Instead of setting the camera lens as a circle, I change the camera's lens to a heart shape, so the blurry reresult will form a heart shape.

## Mesh Loading
I used tinyobj to load the obj file as triangles and do triangle intersection using glm::intersectRayTriangle function.

## Direct lighting
When bounce the ray base on the material BSDF, some of the ray may not hit the light at the end and have no contribution to the final result. So, I take the final ray directly to a random point on an emissive object acting as a light source, to make sure every ray will hit the ligth source and contribute to the final result.

## Removed Terminated Rays With Stream Compaction
After each iteration, I used stream compaction to remove the terminated ray to make more thread available for work.

## Matrial Sorting
Before shade base on the material bsdf, I sorted the ray according to the material type of the ray's intersection point, to make sure the continuous threads are doing the same job so they might finish at the same time.

## First Bounce Caching
For every iteration, when depth is 0, we have to generate ray from the camera first. Instead of generating the ray from camera redundantly for every iteration, I cache the first intersection data and reuse it every time.

## AABB Bounding Box For Mesh Loading
Instead of compute one ray with every triangle of the mesh, I first test the ray with the AABB bounding box of the mesh. If the ray hit the bounding box, then do intersection test with all triangles.



