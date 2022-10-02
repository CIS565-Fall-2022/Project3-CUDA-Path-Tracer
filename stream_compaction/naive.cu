#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernScan(int n, int d, int* odata, int *idata)
        {
            // Inclusive scan

            int k = (blockDim.x * blockIdx.x) + threadIdx.x;

            if (k >= n)
                return;

            int powd = 1 << (d - 1); // More efficient than pow()
            if (k >= powd)
            {
                odata[k] = idata[k - powd] + idata[k];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            
            // Config
            int BLOCK_SIZE = 512;   // This out performs BLOCK_SIZE = 128 in large data sets
            dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
            
            // 2 Buffers
            int* dev_idata;
            int* dev_odata;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Scan            
            for (int d = 1; d <= ilog2ceil(n); ++d)
            {
                kernScan<<<fullBlocksPerGrid, BLOCK_SIZE>>>(n, d, dev_odata, dev_idata);
                checkCUDAError("kernScan fails.");
                cudaMemcpy(dev_idata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);
            }
            
            
            // Inclusive to exclusive (shift right, insert identity)
            cudaMemcpy(odata + 1, dev_odata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            odata[0] = 0;

            cudaFree(dev_idata);
            cudaFree(dev_odata);

            timer().endGpuTimer();
        }
        

        __global__ void kernScan2(int n, int* odata, int* idata)
        {
            // This method should be only used for one block and small data sets

            extern __shared__ int temp[]; // allocated on invocation    
            int thid = threadIdx.x;
            int pout = 0, pin = 1;

            // Load input into shared memory.    
            // This is exclusive scan, so shift right by one
            // and set first element to 0   
            temp[pout * n + thid] = (thid > 0) ? idata[thid - 1] : 0;
            __syncthreads();

            for (int offset = 1; offset < n; offset *= 2)
            {
                pout = 1 - pout; // swap double buffer indices     
                pin = 1 - pout;

                if (thid >= offset)
                    temp[pout * n + thid] += temp[pin * n + thid - offset];
                else
                    temp[pout * n + thid] = temp[pin * n + thid];
                __syncthreads();
            }

            odata[thid] = temp[pout * n + thid]; // write output 
        }

        /**
         * Performs prefix-sum (aka scan) on idata using shared memory, storing the result into odata.
         */
        void scan2(int n, int* odata, const int* idata)
        {
            timer().startGpuTimer();

            if (n > 1024)
            {
                std::cout << "n > 1024, navie scan with shared memory failed." << std::endl;
                timer().endGpuTimer();
                return;
            }

            int* dev_idata;
            int* dev_odata;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            kernScan2<<<1, n, 2 * n * sizeof(int)>>>(n, dev_odata, dev_idata);

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);

            timer().endGpuTimer();
        }
    }
}
