#include <cuda.h>
#include <cuda_runtime.h>
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
        // TODO: __global__

        __global__ void KernShiftToRight(int n,int* odata,int* idata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }
            if (index == 0)
            {
                odata[index] = 0;
            }
            odata[index] = idata[index - 1];
        }

        __global__ void KernRightShiftAddZeros(int* odata, int* middle_buffer, int n, int difference)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n )
            {
                return;
            }
            if (index > (n - 1) - difference)
            {
                odata[index] = 0;
                return;
            }
            odata[index] = middle_buffer[index];
        }


        __global__ void KernNaiveScan(int n,int d,int* odata,const int* idata)
        {
            //for all k in parallel
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }
            //offset: 2^d
            // 2^(offset-1)
            int d_offset = 1 << (d - 1);

            int beginIndex = index - d_offset;
            int prevData = beginIndex >= 0 ? idata[beginIndex] : 0;
            odata[index] = idata[index] + prevData;

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        void scan(int n, int *odata, const int *idata) {
            int blockSize = 256;
     
            //This need to be parallel
            int* dev_idata;
            int* dev_odata;
            int* dev_middleBuffer;

            //allocate memory
            cudaMalloc((void**)&dev_idata, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_middleBuffer, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_middleBuffer failed!");

            cudaDeviceSynchronize();

            //Copy memory from CPU to gpu
            cudaMemcpy(dev_idata,idata,(n)*sizeof(int),cudaMemcpyHostToDevice);
            cudaMemcpy(dev_odata, idata, (n) * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_middleBuffer, idata, (n) * sizeof(int), cudaMemcpyHostToDevice);

            cudaDeviceSynchronize();
            //From host to devicw
            int log2n = ilog2ceil(n);
            int finalMemorySize = 1 << log2n;
            int difference = finalMemorySize - n;


            dim3 BlocksPergrid(finalMemorySize + blockSize - 1 / blockSize);

            timer().startGpuTimer();
            // TODO
            KernRightShiftAddZeros<<<BlocksPergrid, blockSize >>>(dev_idata,dev_middleBuffer,finalMemorySize,difference);
            for (int d = 1; d <= ilog2ceil(finalMemorySize); d++)
            {
                
                KernNaiveScan << <BlocksPergrid, blockSize >> > (finalMemorySize,d,dev_odata,dev_idata);
                cudaDeviceSynchronize();
                //ping pong buffers
                int *dev_temp = dev_idata;
                dev_idata = dev_odata;
                dev_odata = dev_temp;
            }
            KernShiftToRight << <BlocksPergrid, blockSize >> > (finalMemorySize,dev_odata,dev_idata);
            cudaDeviceSynchronize();
         
            
            timer().endGpuTimer();
            //Exclusive scan, so need right shift.

            //copy back to host
            cudaMemcpy(odata , dev_idata, (n) * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_middleBuffer);
        }
    }
}
