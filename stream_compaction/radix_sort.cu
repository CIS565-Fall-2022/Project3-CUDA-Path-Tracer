#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "radix_sort.h"
#include "cpu.h"
#include "efficient.h"
#include <iostream>

#include "common.h"

namespace StreamCompaction 
{
    namespace Radix_Sort 
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __device__ __host__ int checkDigit(int num, int whichDigit)
        {
            return (num >> whichDigit) & 1;
        }

        void radix_sort_cpu(int n, int digitMax, int* odata, const int* idata)
        {
            int* oArray = new int[n];   // A copy of odata for scattering
            int* eArray = new int[n];
            int* fArray = new int[n];
            int* tArray = new int[n];

            // Init odata
            memcpy(odata, idata, n * sizeof(int));

            timer().startCpuTimer();

            for (int i = 0; i < digitMax; ++i)
            {
                // Save the orignal odata
                memcpy(oArray, odata, n * sizeof(int));

                // Build eArray
                for (int j = 0; j < n; ++j)
                {
                    eArray[j] = checkDigit(oArray[j], i) == 1 ? 0 : 1;
                }

                // Build fArray by scaning eArray
                CPU::doScan(n, fArray, eArray);

                // Scatter data by d
                int d;
                int totalFalses = fArray[n - 1] + eArray[n - 1];
                for (int j = 0; j < n; ++j)
                {
                    if (eArray[j] == 0) // b[j] == 1
                    {
                        d = j - fArray[j] + totalFalses; // d[j] = t[j]
                    }
                    else
                    {
                        d = fArray[j];  // d[j] = f[j]
                    }                 
                    odata[d] = oArray[j];
                }
            }

            timer().endCpuTimer();

            delete[] oArray;
            delete[] eArray;
            delete[] fArray;
            delete[] tArray;            
        }




        __global__ void kernBuildErrorArray(int n, int whichOne, int* eArray, int* idata)
        {
            int index = (blockDim.x * blockIdx.x) + threadIdx.x;

            if (index >= n)
                return;

            eArray[index] = 1 - checkDigit(idata[index], whichOne);
        }

        __global__ void kernSplit(int n, int totalFalses, int* odata, int *idata, int *fArray, int* eArray)
        {
            int index = (blockDim.x * blockIdx.x) + threadIdx.x;

            if (index >= n)
                return;

            int d = -1;
            if (eArray[index] == 0)
            {
                d = index - fArray[index] + totalFalses;
            }
            else
            {
                d = fArray[index];
            }

            odata[d] = idata[index];
        }
    
        void radix_sort_parallel(int n, int digitMax, int* odata, int* idata)
        {
            int round_n = 1 << ilog2ceil(n);

            // Configuration
            int BLOCK_SIZE = 128;
            dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int* dev_eArray;
            int* dev_fArray;
            int* dev_odata;
            int* dev_odata2;
            int* tmp = new int[2];

            cudaMalloc((void**)&dev_eArray, round_n * sizeof(int));
            cudaMalloc((void**)&dev_fArray, round_n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata2, n * sizeof(int));
       
            cudaMemset(dev_eArray, 0, round_n * sizeof(int));
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();

            int totalFalses = -1;
            for (int i = 0; i < digitMax; ++i)
            {
                cudaMemcpy(dev_odata2, dev_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);

                // Build eArray
                kernBuildErrorArray<<<fullBlocksPerGrid, BLOCK_SIZE >>>(n, i, dev_eArray, dev_odata2);

                // Build fArray
                cudaMemcpy(dev_fArray, dev_eArray, round_n * sizeof(int), cudaMemcpyDeviceToDevice);
                Efficient::doScan(round_n, dev_fArray);

                // Read totalfalses
                cudaMemcpy(tmp, dev_fArray + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(tmp + 1, dev_eArray + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                totalFalses = tmp[0] + tmp[1];

                // Scatter data
                kernSplit<<<fullBlocksPerGrid, BLOCK_SIZE>>>(n, totalFalses, dev_odata, dev_odata2, dev_fArray, dev_eArray);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            delete[] tmp;
            cudaFree(dev_eArray);
            cudaFree(dev_fArray);
            cudaFree(dev_odata);
            cudaFree(dev_odata2);
        }
    }
}
