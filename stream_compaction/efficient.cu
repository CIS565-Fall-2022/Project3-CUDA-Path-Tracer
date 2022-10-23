#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.h"
#include <iostream>
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep2(int n, int d, int st, int* odata)
        {
            // This uses less threads than kernUpSweep()

            int index = (blockDim.x * blockIdx.x) + threadIdx.x;

            int gap = 1 << d;

            int readIndex = st + 2 * gap * index;
            int writeIndex = readIndex + gap;

            if (writeIndex < n)
            {
                odata[writeIndex] += odata[readIndex];
            }

            __syncthreads();     
        }

        __global__ void kernDownSweep2(int n, int d, int* odata)
        {
            int index = (blockDim.x * blockIdx.x) + threadIdx.x;

            int gap = 1 << d;

            int rightIndex = n - 1 - 2 * gap * index;
            int leftIndex = rightIndex - gap;

            if (leftIndex >= 0)
            {
                int tmp = odata[leftIndex];
                odata[leftIndex] = odata[rightIndex];
                odata[rightIndex] += tmp;
            }

            __syncthreads();
        }

        __global__ void kernUpSweep(int n, int d, int* odata)
        {      
            int k = (blockDim.x * blockIdx.x) + threadIdx.x;

            int powd = 1 << d;
            int powd2 = 1 << (d + 1);

            if (k % powd2 == 0)
            {
                odata[k + powd2 - 1] += odata[k + powd - 1];
            }     
        }

        __global__ void kernDownSweep(int n, int d, int* odata)
        {
            int k = (blockDim.x * blockIdx.x) + threadIdx.x;

            if (k >= n)
                return;

            int powd = 1 << d;
            int powd2 = 1 << (d + 1);

            if (k % powd2 == 0)
            {
                int tmp = odata[k + powd - 1];
                odata[k + powd - 1] = odata[k + powd2 - 1];
                odata[k + powd2 - 1] += tmp;
            }
        }  

        void doScan(int n, int* dev_odata)
        {
            // Exclusive scan

            int round_n = 1 << ilog2ceil(n);

            int BLOCK_SIZE = 128;
            dim3 fullBlocksPerGrid((round_n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            // Up sweep 2 (Halve thread count in each iteration)
            int st = 0;
            int round_n_up_sweep = round_n / 2;
            for (int d = 0; d <= ilog2ceil(round_n) - 1; ++d)
            {
                dim3 fullBlocksPerGridUpSweep((round_n_up_sweep + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernUpSweep2 <<<fullBlocksPerGridUpSweep, BLOCK_SIZE>>> (round_n, d, st, dev_odata);
                checkCUDAError("kernUpSweep fails.");
                st += (1 << d);
                round_n_up_sweep /= 2;
            }
            // Down sweep 2 (Double thread count in each iteration)
            int round_n_down_sweep = 1;
            cudaMemset(dev_odata + round_n - 1, 0, sizeof(int));
            for (int d = ilog2ceil(round_n) - 1; d >= 0; --d)
            {
                dim3 fullBlocksPerGridDownSweep((round_n_down_sweep + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernDownSweep2 <<<fullBlocksPerGridDownSweep, BLOCK_SIZE>>> (round_n, d, dev_odata);
                checkCUDAError("kernDownSweep fails.");
                round_n_down_sweep *= 2;
            }

            //// Up sweep
            //for (int d = 0; d <= ilog2ceil(round_n) - 1; ++d)
            //{                       
            //    kernUpSweep <<<fullBlocksPerGrid, BLOCK_SIZE>>> (round_n, d, dev_odata);
            //    checkCUDAError("kernUpSweep fails.");
            //}
            //// Down sweep
            //cudaMemset(dev_odata + round_n - 1, 0, sizeof(int));
            //for (int d = ilog2ceil(round_n) - 1; d >= 0; --d)
            //{
            //    kernDownSweep << <fullBlocksPerGrid, BLOCK_SIZE >> > (round_n, d, dev_odata);
            //    checkCUDAError("kernDownSweep fails.");
            //}
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
                      
            // Round to 'balanced tree'
            int round_n = 1 << ilog2ceil(n);

            // Device memory setup
            int* dev_odata;
            cudaMalloc((void**)&dev_odata, round_n * sizeof(int));
            cudaMemset(dev_odata, 0, round_n * sizeof(int));
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
  
            timer().startGpuTimer();
            doScan(n, dev_odata);
            timer().endGpuTimer();   

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_odata);
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

            int round_n = 1 << ilog2ceil(n);

            // Config
            int BLOCK_SIZE = 128;
            dim3 fullBlocksPerGrid((round_n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            // Device Memory setup
            int* dev_idata;
            int* dev_odata_bool;
            int* dev_odata_scan;
            
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata_bool, round_n * sizeof(int));
            cudaMalloc((void**)&dev_odata_scan, round_n * sizeof(int));

            cudaMemset(dev_odata_bool, 0, round_n * sizeof(int));
            cudaMemset(dev_odata_scan, 0, round_n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();

            // Mark 1 0
            Common::kernMapToBoolean<<<fullBlocksPerGrid, BLOCK_SIZE>>>(round_n, dev_odata_bool, dev_idata);
            
            // Scan
            cudaMemcpy(dev_odata_scan, dev_odata_bool, n * sizeof(int), cudaMemcpyDeviceToDevice);
            doScan(n, dev_odata_scan);

            // Read total number from scan result
            int* tmp = new int[2];  
            cudaMemcpy(tmp, dev_odata_scan + n - 1, 1 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(tmp + 1, dev_odata_bool + n - 1, 1 * sizeof(int), cudaMemcpyDeviceToHost);
            int count = tmp[0] + tmp[1];
            delete[] tmp;
            
            // Scatter           
            int* dev_odata;
            cudaMalloc((void**)&dev_odata, count * sizeof(int));
            Common::kernScatter <<<fullBlocksPerGrid, BLOCK_SIZE>>>(n, dev_odata, dev_idata, dev_odata_bool, dev_odata_scan);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);

            cudaFree(dev_idata);
            cudaFree(dev_odata_bool);
            cudaFree(dev_odata_scan);
      
            return count;
        }
    }
}
