#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int d, int* odata, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index >= (1 << (d-1))) {
                odata[index] = idata[index - (1 << (d - 1))] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }




        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int* dev_odata;
            int* dev_idata;

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_idata, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            // TODO
            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernNaiveScan <<<fullBlocksPerGrid, blockSize>>> (n, d, dev_odata, dev_idata);
                std::swap(dev_odata, dev_idata);
            }
            timer().endGpuTimer();

            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_idata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_odata);
            cudaFree(dev_idata);
        }
    }
}


