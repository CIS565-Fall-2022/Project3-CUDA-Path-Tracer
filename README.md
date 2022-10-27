CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* XiaoyuDu  
* Tested on: Windows 10, i9-11900KF @ 3.50GHz, RTX 3080 (Personal PC)
  
### Description  
This project built a GPU-based path tracer.
  
### Feature  
I implemented all the features for part 1.  
Below is a comparison between sorting rays by material type and not. I actually find out that sorting by materials make each iterations much slower. My guess is that there are still not so many material types contained in the scene, thus not creating enough branch conditions. Below is a graph that compare the time(in ms) each iteration took with different depth and with sorting by materials ID or not.  
![](./images/1.png)  
Below is a comparison between caching intersections of the first bounce or not.  The graph records the time(in ms) to complete each iteration with different depth.  
![](./images/2.png)  