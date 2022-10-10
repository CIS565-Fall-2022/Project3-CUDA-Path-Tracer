#include "common.h"


//void checkCUDAErrorFn(const char *msg, const char *file, int line) {
//    cudaError_t err = cudaGetLastError();
//    if (cudaSuccess == err) {
//        return;
//    }
//
//    fprintf(stderr, "CUDA error");
//    if (file) {
//        fprintf(stderr, " (%s:%d)", file, line);
//    }
//    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
//    exit(EXIT_FAILURE);
//}

namespace StreamCompaction {
    namespace Common {

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int *bools, const int*idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            bools[index] = idata[index] == 0 ? 0 : 1;
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata,
                const int*idata, const int *bools, const int *indices) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n || bools[index] == 0) {
                return;
            }
            odata[indices[index]] = idata[index];
        }


        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void pathtrace_kernMapToBoolean(int n, int* bools, const PathSegment* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            bools[index] = idata[index].remainingBounces == 0 ? 0 : 1;
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void pathtrace_kernScatter(int n, PathSegment* odata,
            const PathSegment* idata, const int* bools, const int* indices) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n || bools[index] == 0) {
                return;
            }
            //odata[indices[index]].color = idata[index].color;
            //odata[indices[index]].pixelIndex = idata[index].pixelIndex;
            //odata[indices[index]].remainingBounces = idata[index].remainingBounces;
            //odata[indices[index]].ray.direction = idata[index].ray.direction;
            //odata[indices[index]].ray.origin = idata[index].ray.origin;
            memcpy(odata + indices[index], idata + index, sizeof(PathSegment));
            //odata[indices[index]] = idata[index];
        }
    }
}
