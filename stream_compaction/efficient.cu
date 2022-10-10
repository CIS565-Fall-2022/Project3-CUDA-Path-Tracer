#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernUpSweep(int n, int d, int* idata) {
            // Parallel Reduction
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            int k = index * (1 << (d + 1));
            idata[k + (1 << (d + 1)) - 1] += idata[k + (1 << d) - 1];
        }

        __global__ void kernDownSweep(int n, int d, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            int k = index * (1 << (d + 1));
            int t = idata[k + (1 << d) - 1];
            idata[k + (1 << d) - 1] = idata[k + (1 << (d + 1)) - 1];
            idata[k + (1 << (d + 1)) - 1] += t;
        }
       
        __host__ __device__ int copyIlog2(int x) { //copied the given functions bc i am lazy
            int lg = 0;
            while (x >>= 1) {
                ++lg;
            }
            return lg;
        }

        __host__ __device__ int copyIlog2ceil(int x) {
            return x == 1 ? 0 : copyIlog2(x - 1) + 1;
        }
        
        
        // Steps for shared scan
        //    1. Launch kernel with N / (blockSize * 2) blocks, blockSize threads per blockSize
        //    2. For each block generate a shared mem size 2 * blockSize
        //    3. Load values from input array in pairs to shared mem
        //    4. Do same indexing upsweep scheme as before on individual blocks
        //    5. Get INCLUSIVE element for each block endand add to temp array
        //    6. Zero out last element of each block, like with root zeroing
        //    7. Down sweep on individual blocks
        //      OR DO REST ON CPU :)
        //    8. Pass the temp array as the new input array and recurse steps 1 - 7
        //    9. Recursively add the output of the temp array as an offset to each block
        __global__ void kernSharedScan(int n, int* idata, int* temp) {
            // Parallel Reduction w/ shared memory
            // Shared memory should be 2 * blockSize
            __shared__ int partialSum[2 * blockSize];
            // Load input memory into shared memory in pairs
            int index = threadIdx.x + (blockIdx.x * blockDim.x); //index of all launched threads (N / 2)
            int sharedIdx = threadIdx.x; // per block index
            partialSum[sharedIdx * 2] = idata[index * 2];
            partialSum[sharedIdx * 2 + 1] = idata[index * 2 + 1];
            // Per block upsweep
            int logBlock = copyIlog2ceil(blockDim.x * 2); //blockSize * 2 since we are doing blockSize*2 elements per block
            for (int d = 0; d < logBlock; ++d) { // Runs log2(blockSize) times
                __syncthreads();
                if (sharedIdx < (blockDim.x / (1 << d))) {
                    int k = sharedIdx * (1 << (d + 1));
                    partialSum[k + (1 << (d + 1)) - 1] += partialSum[k + (1 << d) - 1];
                }
            }
            __syncthreads();
            // Save last INCLUSIVE VALUE of block (for recursion and offset)
            // Zero out root
            if (sharedIdx == blockDim.x - 1) { // Last thread in block
                temp[blockIdx.x] = partialSum[2 * blockDim.x - 1]; //+ idata[(2 * blockDim.x - 1) + blockIdx.x * blockDim.x * 2]; // Last element in shared mem + last element of block in idata (last inclusive element)
                partialSum[2 * blockDim.x - 1] = 0; 
            }
            __syncthreads();
            // Per block downsweep
            for (int d = logBlock - 1; d >= 0; --d) {
                if (sharedIdx < (blockDim.x / (1 << d))) {
                    int k = sharedIdx * (1 << (d + 1));
                    int t = partialSum[k + (1 << d) - 1];
                    partialSum[k + (1 << d) - 1] = partialSum[k + (1 << (d + 1)) - 1];
                    partialSum[k + (1 << (d + 1)) - 1] += t;
                }
            }
            __syncthreads();
            //Write to input array in place
            idata[index * 2] = partialSum[sharedIdx * 2];
            idata[index * 2 + 1] = partialSum[sharedIdx * 2 + 1];
        }

        // Function to add offset buffer to each block
        // ex. offset = [10, 20, 30], add 10 to block 0, add 20 to block 1, add 30 to block 2   
        __global__ void addOffsets(int n, int* idata, int* offsets) {
            // n is num elements in idata
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index > n) {
                return;
            }
            idata[index] += offsets[(int)index / (blockDim.x * 2)];
        }

        void scan(int n, int* odata, const int* idata) {
                int paddedN = (1 << ilog2ceil(n));
                int* dev_idata;
                cudaMalloc((void**)&dev_idata, paddedN * sizeof(int));
                //checkCUDAError("cudaMalloc dev_idata failed!");
                cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaMemset(dev_idata + n, 0, (paddedN - n) * sizeof(int));
                //cudaDeviceSynchronize();

                //timer().startGpuTimer();
                //Determine size of temp array after 1 pass
                int tempSize = n / (blockSize * 2);
                dim3 gridSize(tempSize);
                int* dev_temp;
                int* temp = (int*) malloc(tempSize * sizeof(int));
                cudaMalloc((void**)& dev_temp, tempSize * sizeof(int));
                cudaDeviceSynchronize();
                kernSharedScan << <gridSize, blockSize >> > (paddedN, dev_idata, dev_temp);
                //checkCUDAError("kernSharedScan failed!");
                cudaDeviceSynchronize();
                cudaMemcpy(temp, dev_temp, tempSize * sizeof(int), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                int prev = 0;
                for (int i = 0; i < tempSize; ++i) { // In-place CPU exclusive scan
                    int tempVal = temp[i];
                    temp[i] = prev;
                    prev += tempVal;
                }
                
                cudaMemcpy(dev_temp, temp, tempSize * sizeof(int), cudaMemcpyHostToDevice);
                cudaDeviceSynchronize();
                dim3 offsetGridSize(paddedN / blockSize);
                addOffsets << <offsetGridSize, blockSize >> > (paddedN, dev_idata, dev_temp);
                //checkCUDAError("addOffsets failed!");
                cudaDeviceSynchronize();
                //timer().endGpuTimer();

                cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
                cudaFree(dev_idata);
                cudaFree(dev_temp);
                free(temp);
        }

        __global__ void kernZeroRoot(int n, int* idata) {
            // Root is last element
            idata[n - 1] = 0;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void oldScan(int n, int *odata, const int *idata) {
            // Account for non-powers of 2 by padding by 0
            int paddedN = (1 << ilog2ceil(n));
            int* dev_idata;
            cudaMalloc((void**)&dev_idata, paddedN * sizeof(int));
            //checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_idata + n, 0, (paddedN - n) * sizeof(int));
            cudaDeviceSynchronize();

            timer().startGpuTimer();
            // Upsweep
            for (int i = 0; i < ilog2ceil(n); ++i) {
                int numThreads = paddedN / (1 << (i + 1));
                dim3 upSweepGridSize((numThreads + blockSize - 1) / blockSize);
                kernUpSweep << <upSweepGridSize, blockSize >> >
                    (numThreads, i, dev_idata);
                //checkCUDAError("kernUpSweep failed!");
                cudaDeviceSynchronize();
            }

            // Downsweep
            kernZeroRoot << <1, 1 >> > (paddedN, dev_idata);
            for (int i = ilog2ceil(n) - 1; i >= 0; --i) {
                int numThreads = paddedN / (1 << (i + 1));
                dim3 downSweepGridSize((numThreads + blockSize - 1) / blockSize);
                kernDownSweep << <downSweepGridSize, blockSize >> >
                    (numThreads, i, dev_idata);
                //checkCUDAError("kernDownSweep failed!");
                cudaDeviceSynchronize();
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
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
        int compact(int n, PathSegment* dev_odata, const PathSegment* dev_idata) {
            // Account for non-powers of 2 by padding by 0
            int paddedN = (1 << ilog2ceil(n));
            //int* dev_idata;
            //int* dev_odata;
            int* dev_bool;
            int* dev_indices;

            //cudaMalloc((void**)&dev_idata, n * sizeof(PathSegment));
            //checkCUDAError("cudaMalloc dev_idata failed!");
            //cudaMemcpy(dev_idata, idata, n * sizeof(PathSegment), cudaMemcpyHostToDevice);
            
            // Pad bool array instead of idata to save operations in kernMapToBoolean
            cudaMalloc((void**)&dev_bool, paddedN * sizeof(int));
            //checkCUDAError("cudaMalloc dev_bool failed!");
            cudaMemset(dev_bool + n, 0, (paddedN - n) * sizeof(int));

            cudaMalloc((void**)&dev_indices, paddedN * sizeof(int));
            //checkCUDAError("cudaMalloc dev_indices failed!");
            //cudaMalloc((void**)&dev_odata, n * sizeof(int));
            //checkCUDAError("cudaMalloc dev_indices failed!");
            cudaDeviceSynchronize();


            //Determine size of temp array after 1 pass
            int tempSize = paddedN / (blockSize * 2);
            dim3 gridSize(tempSize);
            int* dev_temp;
            int* temp = (int*)malloc(tempSize * sizeof(int));
            cudaMalloc((void**)&dev_temp, tempSize * sizeof(int));
            //checkCUDAError("cudaMalloc dev_temp failed!");

            //timer().startGpuTimer();
            // Binarize
            dim3 nGridSize((n + blockSize - 1) / blockSize);
            StreamCompaction::Common::pathtrace_kernMapToBoolean << < nGridSize, blockSize >> >
                (n, dev_bool, dev_idata);
            //checkCUDAError("kernMapToBoolean failed!");
            cudaDeviceSynchronize();
            // We need bool array for scatter so copy bool result to indices to be modified in place
            cudaMemcpy(dev_indices, dev_bool, paddedN * sizeof(int), cudaMemcpyDeviceToDevice);
            //checkCUDAError("cudaMemcpy failed!");
            cudaDeviceSynchronize();
            
            // Shared scan copied from above
            kernSharedScan << <gridSize, blockSize >> > (paddedN, dev_indices, dev_temp);
            //checkCUDAError("kernSharedScan failed!");
            cudaDeviceSynchronize();
            cudaMemcpy(temp, dev_temp, tempSize * sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            int prev = 0;
            for (int i = 0; i < tempSize; ++i) { // In-place CPU exclusive scan
                int tempVal = temp[i];
                temp[i] = prev;
                prev += tempVal;
            }
            cudaMemcpy(dev_temp, temp, tempSize * sizeof(int), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            dim3 offsetGridSize(paddedN / blockSize);
            addOffsets << <offsetGridSize, blockSize >> > (paddedN, dev_indices, dev_temp);
            //checkCUDAError("addOffsets failed!");
            cudaDeviceSynchronize();
            //// Copied Scan code from above
            //// Upsweep
            //for (int i = 0; i < ilog2ceil(n); ++i) {
            //    int numThreads = paddedN / (1 << (i + 1));
            //    dim3 upSweepGridSize((numThreads + blockSize - 1) / blockSize);
            //    kernUpSweep << <upSweepGridSize, blockSize >> >
            //        (numThreads, i, dev_indices);
            //    checkCUDAError("kernUpSweep failed!");
            //    cudaDeviceSynchronize();
            //}

            // Downsweep
            //kernZeroRoot << <1, 1 >> > (paddedN, dev_indices);
            //for (int i = ilog2ceil(n) - 1; i >= 0; --i) {
            //    int numThreads = paddedN / (1 << (i + 1));
            //    dim3 downSweepGridSize((numThreads + blockSize - 1) / blockSize);
            //    kernDownSweep << <downSweepGridSize, blockSize >> >
            //        (numThreads, i, dev_indices);
            //    checkCUDAError("kernDownSweep failed!");
            //    cudaDeviceSynchronize();
            //}

            // Scatter
            StreamCompaction::Common::pathtrace_kernScatter << <nGridSize, blockSize >> >
                (n, dev_odata, dev_idata, dev_bool, dev_indices);
            //checkCUDAError("kernScatter failed!");
            cudaDeviceSynchronize();
            //timer().endGpuTimer();
            int* finalNum = (int*) malloc(sizeof(int));
            cudaMemcpy(finalNum, dev_indices + paddedN - 1, sizeof(int), cudaMemcpyDeviceToHost);
            //cudaMemcpy(odata, dev_odata, finalNum * sizeof(int), cudaMemcpyDeviceToHost);
            //cudaFree(dev_idata);
            cudaFree(dev_bool);
            cudaFree(dev_indices);
            //cudaFree(dev_odata);
            return finalNum[0];
        }
    }
}
