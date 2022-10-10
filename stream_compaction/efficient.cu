#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <device_launch_parameters.h>
#include <device_functions.h>


#define NUM_BANKS 16 
#define LOG_NUM_BANKS 4 
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void prescan(float* g_odata, float* g_idata, int n) {
          extern __shared__ float temp[];  // allocated on invocation 
          int thid = threadIdx.x; int offset = 1; 
          
          int ai = thid; int bi = thid + (n / 2); 
          int bankOffsetA = CONFLICT_FREE_OFFSET(ai); 
          int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
          temp[ai + bankOffsetA] = g_idata[ai];
          temp[bi + bankOffsetB] = g_idata[bi];
            
          for (int d = n >> 1; d > 0; d >>= 1)                    // build sum in place up the tree 
          {
            __syncthreads();
            if (thid < d) {
              int ai = offset * (2 * thid + 1) - 1; 
              int bi = offset * (2 * thid + 2) - 1; 
              ai += CONFLICT_FREE_OFFSET(ai);
              bi += CONFLICT_FREE_OFFSET(bi);

              temp[bi] += temp[ai];
            }

            offset *= 2;
          }

          if (thid == 0) { temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; }

          for (int d = 1; d < n; d *= 2) // traverse down tree & build scan 
          {      
            offset >>= 1;      
            __syncthreads();      
            
            if (thid < d) { 
              int ai = offset * (2 * thid + 1) - 1;     
              int bi = offset * (2 * thid + 2) - 1;

              float t = temp[ai]; temp[ai] = temp[bi]; temp[bi] += t;

            } 
          }  
          
          __syncthreads();
          g_odata[ai] = temp[ai + bankOffsetA]; 
          g_odata[bi] = temp[bi + bankOffsetB];
        }

        __global__ void kernAdd

        __global__ void kernUpSweep(int N, int offset, int* idata) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= N)
            return;

          int idx = index * offset;
          idata[idx + offset - 1] += idata[idx + offset / 2 - 1];
        }

        __global__ void kernDownSweep(int N, int offset, int* idata) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= N)
            return;

          int idx = index * offset;
          int t = idata[idx + offset / 2 - 1];
          idata[idx + offset / 2 - 1] = idata[idx + offset - 1];
          idata[idx + offset - 1] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
            int N = pow(2, ilog2ceil(n));
            int logN = ilog2ceil(n);
            int* dataGPU;
            cudaMalloc((void**)&dataGPU, N * sizeof(int));

            cudaMemcpy(dataGPU, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            /* Up Sweep */
            int num = N / 2, offset = 2;
            for (; num > 0; num /= 2, offset *= 2) {
              dim3 fullBlocksPerGrid((num + blockSize - 1) / blockSize);
              kernUpSweep << <fullBlocksPerGrid, blockSize >> > (num, offset, dataGPU);
            }
            
            /* Down Sweep */
            int zero = 0;
            offset /= 2;
            num = 1;
            cudaMemcpy(dataGPU + (N - 1), &zero, sizeof(int), cudaMemcpyHostToDevice);
            for (; num < N; num *= 2, offset /= 2) {
              dim3 fullBlocksPerGrid((num + blockSize - 1) / blockSize);
              kernDownSweep << <fullBlocksPerGrid, blockSize >> > (num, offset, dataGPU);
            }
            
            timer().endGpuTimer();

            cudaMemcpy(odata, dataGPU, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dataGPU);


        }

        __global__ void kernResetIntBuffer(int N, int* intBuffer, int value) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index < N) {
            intBuffer[index] = value;
          }
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
            
            // TODO          
            int N = pow(2, ilog2ceil(n)); 
            dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

            int* dev_odata;
            int* dev_idata;
            int* dev_boolData;
            int* dev_indicesData;
            cudaMalloc((void**)&dev_idata, N * sizeof(int));
            //checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");
            cudaMalloc((void**)&dev_odata, N * sizeof(int));
            //checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");
            cudaMalloc((void**)&dev_boolData, N * sizeof(int));
            //checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");
            cudaMalloc((void**)&dev_indicesData, N * sizeof(int));
            //checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

            kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (N, dev_boolData, 0);

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            /* Start Timer */
            timer().startGpuTimer();

            StreamCompaction::Common::kernMapToBoolean << < fullBlocksPerGrid, blockSize >> > (n, dev_boolData, dev_idata);

            
            /* Scan */
            cudaMemcpy(dev_indicesData, dev_boolData, N * sizeof(int), cudaMemcpyDeviceToDevice);
            /* Up Sweep */
            int num = N / 2, offset = 2;
            for (; num > 0; num /= 2, offset *= 2) {
              dim3 fullBlocksPerGrid((num + blockSize - 1) / blockSize);
              kernUpSweep << <fullBlocksPerGrid, blockSize >> > (num, offset, dev_indicesData);
            }

            /* Down Sweep */
            int zero = 0;
            offset /= 2;
            num = 1;
            cudaMemcpy(dev_indicesData + (N - 1), &zero, sizeof(int), cudaMemcpyHostToDevice);
            for (; num < N; num *= 2, offset /= 2) {
              dim3 fullBlocksPerGrid((num + blockSize - 1) / blockSize);
              kernDownSweep << <fullBlocksPerGrid, blockSize >> > (num, offset, dev_indicesData);
            }

            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (N, dev_odata, dev_idata, dev_boolData, dev_indicesData);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaMemcpy(&num, dev_indicesData + (N - 1), sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_boolData);
            cudaFree(dev_indicesData);


            return num;
        }

        __global__ void kernRadixSort(int n, int* data, int bit) {
          __shared__ float sM[TILE_WIDTH];

          int tx = threadIdx.x;
          int index = (blockIdx.x * TILE_WIDTH) + threadIdx.x;
          if (index >= n) {
            return;
          }

          sM[tx] = data[index];
          __syncthreads();
        }

        __global__ void kernBitOperation(int N, int bit, int* odata, int* idata) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= N)
            return;

          odata[index] = ((idata[index] & (1 << bit)) == 0);
        }

        __global__ void kernSplit(int N, int bit, int totalFalses, int* odata, int* idata, int* falseIndices) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= N)
            return;

          int inValue = idata[index];
          int falseIndex = falseIndices[index];
          if ((inValue & (1 << bit)) != 0) {
            int trueIndex = index - falseIndex + totalFalses;
            odata[trueIndex] = inValue;
          }
          else {
            odata[falseIndex] = inValue;
          }
          
        }

        __global__ void kernFindMax(int n, int* idata) {
          int index = (blockIdx.x * blockDim.x) + threadIdx.x;
          if (index >= n)
            return;

          int curr = idata[index], next = idata[index + n];
          idata[index] = (curr > next) ? curr : next;
        }

        int max(int n, int* idata) {
          int num = n / 2, offset = 2;
          for (; num > 0; num /= 2, offset *= 2) {
            dim3 fullBlocksPerGrid((num + blockSize - 1) / blockSize);
            kernFindMax << <fullBlocksPerGrid, blockSize >> > (num, idata);
          }

          cudaMemcpy(&num, idata, sizeof(int), cudaMemcpyDeviceToHost);
          return num;
        }


        /**
         * Performs radix sort on idata, storing the result into odata.
         * The result array is sorted
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to sort.
         * @returns      The number of elements remaining after compaction.
         */
        void radixSort(int n, int* odata, const int* idata) {
          int N = pow(2, ilog2ceil(n));
          dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

          int* dev_odata;
          int* dev_idata;
          int* dev_indicesData;
          cudaMalloc((void**)&dev_idata, N * sizeof(int));
          cudaMalloc((void**)&dev_odata, N * sizeof(int));
          cudaMalloc((void**)&dev_indicesData, N * sizeof(int));

          cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
          cudaMemset(dev_odata + n, INT_MIN, (N - n) * sizeof(int));

          cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

          
          timer().startGpuTimer();
          int dev_max = StreamCompaction::Efficient::max(N, dev_odata);
          int maxBits = ilog2ceil(dev_max);

          for (int bit = 0; bit <= maxBits; bit++) {
            int lastBool, lastScan;
            kernBitOperation << <blocksPerGrid, blockSize >> > (n, bit, dev_indicesData, dev_idata);

            cudaMemcpy(&lastBool, dev_indicesData + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);

            /* Scan */
            //cudaMemset(dev_indicesData + n, 0, (N - n) * sizeof(int));

            /* Up Sweep */
            int num = N / 2, offset = 2;
            for (; num > 0; num /= 2, offset *= 2) {
              dim3 fullBlocksPerGrid((num + blockSize - 1) / blockSize);
              kernUpSweep << <fullBlocksPerGrid, blockSize >> > (num, offset, dev_indicesData);
            }

            /* Down Sweep */
            int zero = 0;
            offset /= 2;
            num = 1;
            cudaMemcpy(dev_indicesData + (N - 1), &zero, sizeof(int), cudaMemcpyHostToDevice);
            for (; num < N; num *= 2, offset /= 2) {
              dim3 fullBlocksPerGrid((num + blockSize - 1) / blockSize);
              kernDownSweep << <fullBlocksPerGrid, blockSize >> > (num, offset, dev_indicesData);
            }

            cudaMemcpy(&lastScan, dev_indicesData + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);

            int totalFalses = lastBool + lastScan;
            kernSplit << <blocksPerGrid, blockSize >> > (n, bit, totalFalses, dev_odata, dev_idata, dev_indicesData);

            std::swap(dev_odata, dev_idata);
          }
          timer().endGpuTimer();

          cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

          cudaFree(dev_idata);
          cudaFree(dev_odata);
          cudaFree(dev_indicesData);
        }
    }
}

