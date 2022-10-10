#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <device_launch_parameters.h>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernScan(int N, int offset, int* odata, int* idata) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= N)
            return;

          int inValue = idata[index];
          if (index >= offset) 
            odata[index] = idata[index - offset] + inValue;
          else
            odata[index] = inValue;
          
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int offset = 1;
            int logN = ilog2ceil(n);
            int* dev_odata;
            int* dev_idata;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            /* Start Timer */
            timer().startGpuTimer();

            for (int d = 0; d < logN; d++, offset *= 2) {
              std::swap(dev_idata, dev_odata);
              kernScan << <fullBlocksPerGrid, blockSize >> > (n, offset, dev_odata, dev_idata);
            }

            timer().endGpuTimer();

            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_odata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
