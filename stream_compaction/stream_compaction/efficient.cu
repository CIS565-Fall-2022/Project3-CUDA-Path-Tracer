#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        __global__ void kernUpSweep(int n, int d, int* x) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index % (1 << (d + 1)) == 0) {
                x[index + (1 << (d + 1)) - 1] += x[index + (1 << d) - 1];
            }
        }

        __global__ void kernDownSweep(int n, int d, int* x) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index % (1 << (d + 1)) == 0) {
                int t = x[index + (1 << d) - 1];
                x[index + (1 << d)  - 1] = x[index + (1 << (d+1)) - 1];
                x[index + (1 << (d + 1)) - 1] = t + x[index + (1 << (d + 1)) - 1];
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int* dev_idata;
            int size = 1 << ilog2ceil(n);

            cudaMalloc((void**)&dev_idata, size * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            // TODO

            for (int d = 0; d < ilog2ceil(size); d++) {
                cudaDeviceSynchronize();
                kernUpSweep <<<fullBlocksPerGrid, blockSize>>> (n, d, dev_idata);
            }

            cudaDeviceSynchronize();
            cudaMemset(dev_idata + size - 1, 0, sizeof(int));

            for (int d = ilog2ceil(size) - 1; d >= 0; d--) {
                cudaDeviceSynchronize();
                kernDownSweep <<<fullBlocksPerGrid, blockSize>>> (n, d, dev_idata);
            }


            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, n  * sizeof(int), cudaMemcpyDeviceToHost);\

            cudaFree(dev_idata);
        }





        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int* dev_odata;
            int* dev_idata;
            int* dev_indices;
            int* dev_bools;
            int size = 1 << ilog2ceil(n);
            int retSize = 0;

            cudaMalloc((void**)&dev_idata, size * sizeof(int));
            cudaMalloc((void**)&dev_odata, size * sizeof(int));
            cudaMalloc((void**)&dev_bools, size * sizeof(int));
            cudaMalloc((void**)&dev_indices, size * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            dim3 fullBlocksPerGridCeil((size + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            // TODO
            Common::kernMapToBoolean <<<fullBlocksPerGrid, blockSize>>> (n, dev_bools, dev_idata);
            cudaDeviceSynchronize();
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);

            for (int d = 0; d < ilog2ceil(size); d++) {
                kernUpSweep <<<fullBlocksPerGridCeil, blockSize >>> (n, d, dev_indices);
                cudaDeviceSynchronize();
            }

            cudaMemset(dev_indices + size - 1, 0, sizeof(int));

            for (int d = ilog2ceil(size) - 1; d >= 0; d--) {
                kernDownSweep <<<fullBlocksPerGridCeil, blockSize>>> (n, d, dev_indices);
                cudaDeviceSynchronize();
            }

            Common::kernScatter <<<fullBlocksPerGrid, blockSize>>> (n, dev_odata, dev_idata, dev_bools, dev_indices);
            
            timer().endGpuTimer();
            cudaMemcpy(&retSize, dev_indices + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_odata, retSize * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);
            cudaFree(dev_idata);
           
            return retSize;
        }
    }
}
