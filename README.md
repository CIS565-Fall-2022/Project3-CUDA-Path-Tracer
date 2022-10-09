CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* David Li
* [LinkedIn](https://www.linkedin.com/in/david-li-15b83817b/)
* Tested on: Windows 10, Intel(R) Core(TM) i9-10980HK CPU @ 2.40GHz 32Gb, GTX 2070 Super (Personal Computer)

[Repo Link](https://github.com/theCollegeBoardOfc/Project3-CUDA-Path-Tracer)

### Why Path Trace?

The primary motivation behind path tracing is to properly render light in a scene. This is done essentially by shooting several rays into a scene, letting them bounce around and accumulate color. If however, this is only performed once, the render looks rather poor, thus we instead will repeat the process hundreds sometimes thousands of times and take an average of all those colors to yield a final sample. This takes a rather long time: an 800x800 resolution image would have 640000 pixels, we would shoot out at least one ray (sometimes more) per pixel for each sample, if we take 5000 samples, then the path tracer would be shooting out 3.2 billion rays. This rendered scene shoots out four times more rays than that.

![](img/angel.png)

This scene is rendered with around 1000 samples.

![](img/discus1.png)

### "Embarassingly Parallel"

Perhaps the one saving grace about all those rays is that much of this process can be parallelized. The color accumulated by one ray does not depend on the color of another. The CPU can only parallelize a small quantity of processes, capped by number of cores on the CPU. The GPU on the other hand was designed for this kind of work. Thousands of processes can be run on the GPU at once, making that 3.2 billion figure a lot less daunting.

### Stream Compaction

Simply developing the path-tracer on the GPU rather the CPU yields an incredible increase in performance, but we can still do better. The first major step in improving performance is to do stream compaction on terminated rays. In order to properly simulate light, every single ray bounces around the scene until one of three termination conditions. 1. It hits a light. 2. It hits nothing. 3. It reaches its maxdepth. Hitting the max depth is the latest a ray will terminate. It's basically telling the ray, "Ok you've gone on long enough, time to just tell me what you have and move on." Going to max depth for a ray is the worst case scenario, we would much rather the ray not hit anything or hit a light, these are earlier termination conditions, and are the reason stream compaction works. For a single sample, here is the effects of stream compaction. 

![](img/chart1.png)

As the depth increases, we have to do less and less work because stream compaction is removing terminated rays. Also, note that there are zero rays at depth 8, this was the max depth, so all rays were terminated. The next chart compares a closed system and an open system. The open system has no wall behind the scene, meaning ray can terminate by bouncing out and hitting nothing. The closed system has a wall behind the scene, meaning any ray that would have bounced out are reflected back into the scene.

![](img/chart2.png)

As is likely expected: the closed system benefits much less from stream compaction. One of ray termination condtions, not hitting anything, will never be satisfied under the closed system. 

### Other performance "improvements"
Another "improvement" that was introduced was sorting by material type. In other words, when the rays intersect an object that object has a material type which encompasses objects color, emittance and how rays should bounce off it. If we sort by material type, then we leverage the architecture of the GPU to be more efficient. This is because processes or threads are executed in groups of 32: called a warp. If just one thread in a warp is running, then the other 31 threads just sit idle. By sorting by material, we minimize the chance of this sort of thing happening with the idea that rays with the same material type will do a similar amount of work and terminate at around the same time. Unfortunately, this did not really work out, on running a test scene the frame per second on the render dropped ever so slightly when sorting by material which is bad. This is likely because the work of sorting by materials takes longer than the amount of time saved by the sorting. 



