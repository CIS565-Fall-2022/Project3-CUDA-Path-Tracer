CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Eyad Almoamen
  * [LinkedIn](https://www.linkedin.com/in/eyadalmoamen/), [personal website](https://eyadnabeel.com)
* Tested on: Windows 11, i7-10750H CPU @ 2.60GHz 2.59 GHz 16GB, RTX 2070 Super Max-Q Design 8GB (Personal Computer)

Introduction
================
I've built a GPU accelerated monte carlo path tracer using CUDA and C++. Features include [TODO: ADD FEATURE DESCRIPTION].
The parallelization is happening on a ray-by-ray basis, with the terminated rays being eliminated via stream compaction and sorted by material type in order to avoid warp divergence.
