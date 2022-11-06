#ifdef __INTELLISENSE__
#define KERN_PARAM(x,y)
#define __syncthreads()
#include <device_launch_parameters.h>
#else
#define __syncthreads() __syncthreads()
#define KERN_PARAM(x,y) <<< x,y >>>
#endif