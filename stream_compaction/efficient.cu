#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        using StreamCompaction::Common::kernMapToBoolean;
        using StreamCompaction::Common::kernScatter;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
#define blockSize 256

        __global__ void KernUpSweep(int n, int* data,int d)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            //real offset is 2^power
            if (index >= n)
                return;
            int pow1 = 1 << (d + 1);
            int pow2 = 1 << d;
            if (index % pow1 == 0)
            {
                data[index + pow1 - 1] += data[index + pow2 - 1];
            }
        }

        __global__ void KernDownSweep(int n,int* data,int d)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
                return;
            int pow1 = 1 << (d + 1);
            int pow2 = 1 << d;
            if (index % pow1 == 0)
            {
                int t = data[index + pow2 - 1];
                data[index + pow2 - 1] = data[index + pow1 - 1];
                data[index + pow1 - 1] += t;
            }
        }


        //set n-1 =0
       
        __global__ void KernSetZero(int n,int* idata)
        {
            idata[n - 1] = 0;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        void scan(int n, int *odata, const int *idata, bool gpuTimerStart) {
            int* dev_data;
            int* dev_buffer;

            int log2n = ilog2ceil(n);
            //input array may not be two power 
            //So need to get nearest two power
            int nearest_2power = 1 << log2n;
            int finalMemorySize = nearest_2power;
            int difference =  finalMemorySize-n;

            dim3 fullBlocksPergrid((finalMemorySize + blockSize - 1) / blockSize);

            //allocate cuda memoty
            cudaMalloc((void**)&dev_data, finalMemorySize * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMalloc((void**)&dev_buffer, finalMemorySize * sizeof(int));
            checkCUDAError("cudaMemset dev_buffer failed!");

            cudaMemset(dev_data, 0, finalMemorySize * sizeof(int));
            checkCUDAError("cudaMemset dev_data failed!");

            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_data failed!");

            cudaMemcpy(dev_buffer, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_buffer failed!");

            if (gpuTimerStart == false)
            {
                timer().startGpuTimer();
            }
       
            // TODO
            int real_d = ilog2ceil(finalMemorySize);
            //upsweep
            for (int d = 0; d <= real_d - 1; d++)
            {
                KernUpSweep << <fullBlocksPergrid, blockSize >> > (finalMemorySize,dev_data,d);
                checkCUDAError("KernupSweep failed!");
                
            }
            //down Sweep
            KernSetZero << < 1, 1 >> > (finalMemorySize, dev_data);
            for (int d = real_d - 1; d >= 0; d--)
            {
                KernDownSweep << <fullBlocksPergrid, blockSize >> > (nearest_2power,dev_data,d);
                checkCUDAError("KernDownSweep failed!");
            }
           
            if (gpuTimerStart == false)
            {
                timer().endGpuTimer();
            }
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
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
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            int* count = new int[2];

            int* dev_idata;
            int* dev_odata;
            int* dev_bool;
            int* dev_boolScan;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("CUDA Malloc dev_idata failed!");
            cudaMalloc((void**)&dev_bool, n * sizeof(int));
            checkCUDAError("CUDA Malloc dev_bool failed!");
            cudaMalloc((void**)&dev_boolScan, n * sizeof(int));
            checkCUDAError("CUDA Malloc dev_boolScan failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("CUDA Malloc dev_odata failed!");

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bool, dev_idata);
            checkCUDAError("kernMapToBoolean failed!");

            scan(n, dev_boolScan, dev_bool,true);

            kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bool, dev_boolScan);
            checkCUDAError("kernScatter failed!");

            timer().endGpuTimer();

            cudaMemcpy(count, dev_bool + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cudaMemcpy(count+1, dev_boolScan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            //size equals to last of boolean array and last of boolean prefix sum array
            int compactedSize = count[0] + count[1];

            cudaMemcpy(odata, dev_odata, sizeof(int) * compactedSize, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy back failed!");

            cudaFree(dev_idata);
            cudaFree(dev_bool);
            cudaFree(dev_boolScan);
            cudaFree(dev_odata);

            return compactedSize;
        }
    }
}
