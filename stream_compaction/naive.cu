#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <device_launch_parameters.h>
#define blockSize 256

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        //// DEBUGGER TEST
        //__global__ void kernTestDebugger(int param) {
        //    int index = threadIdx.x + (blockIdx.x * blockDim.x);
        //    index = 1;
        //    index = threadIdx.x + (blockIdx.x * blockDim.x);
        //    param = index;
        //}

        __global__ void kernNaiveScan(int n, int d, int offset, int *odata, const int *idata) {
            int k = threadIdx.x + blockIdx.x * blockDim.x;
            if (k >= n) {
                return;
            }
            
            if (k >= offset) {
                odata[k] = idata[k - offset] + idata[k];
            }
            else {
                odata[k] = idata[k];
            }

        }

        __global__ void kernInclusiveToExclusive(int n, int* odata, const int* idata) {
            // shift all elements right and keep 1st element as identity 0
            int k = threadIdx.x + blockIdx.x * blockDim.x;
            if (k >= n) {
                return;
            }
            if (k == 0) {
                odata[k] = 0;
            }
            else {
                odata[k] = idata[k - 1];
            }
        }
        
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            

            //// DEBUGGER TEST
            // int noOfBlocks = 1;
            // dim3 blockSize(32, 32);
            // kernTestDebugger << < noOfBlocks, blockSize >> > (2);
            // 
            
            int* dev_buffer1;
            int* dev_buffer2;

            /*dim3 gridSize(32, 32);
            dim3 blockSize(32, 32);*/

            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

            // Memory allocation
            cudaMalloc((void**)&dev_buffer1, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_buffer1 failed!");
            cudaMalloc((void**)&dev_buffer2, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_buffer2 failed!");
            cudaMemcpy(dev_buffer1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("memcpy into dev_buffer1 failed!");

            

            int maxDepth = ilog2ceil(n);

            timer().startGpuTimer();
            for (int d = 1; d <= maxDepth; d++) {    // where d is depth of iteration
                int offset = pow(2, d - 1);
                kernNaiveScan << <blocksPerGrid, blockSize >> > (n, d, offset, dev_buffer2, dev_buffer1);
                cudaMemcpy(dev_buffer1, dev_buffer2, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            }
            // converting from inclusive to exclusive scan using same buffers
            kernInclusiveToExclusive << <blocksPerGrid, blockSize >> > (n, dev_buffer1, dev_buffer2);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_buffer1, sizeof(int) * (n), cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy into odata failed!");

            cudaFree(dev_buffer1);
            cudaFree(dev_buffer2);

        }
    }
}
