#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <device_launch_parameters.h>
#define blockSize 256
namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /*
         * Kernel for parallel reduction with upstream scan
         */
        __global__ void kernUpSweepReduction(int n, int d, int offsetd, int offsetd1, int* data) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }
            if (k % offsetd1 == 0) {
                data[k + offsetd1 - 1] = data[k + offsetd1 - 1] + data[k + offsetd - 1];
                return;
            }

            //// Tried implementing optimized upsweep
            //if (k < offsetd) {
            //    data[k] += data[k + offsetd];
            //}
        }

        /*
         * Kernel for collecting results with downsweep scan
         */
        __global__ void kernDownSweep(int n, int d, int offsetd, int offsetd1, int* data) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }
            if (k % offsetd1 != 0) {
                return;
            }
            int t = data[k - 1 + offsetd];               // Save left child
            data[k - 1 + offsetd] = data[k - 1 + offsetd1];  // Set left child to this node’s value
            data[k - 1 + offsetd1] += t;
        }

        /*
         * Kernel to parallelly map input data to 0 and 1 based on whether
         * it meets criteria for stream compaction
         */
        __global__ void kernMap(int n, int* odata, const int* idata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }
            odata[k] = (idata[k] == 0) ? 0 : 1;
        }

        /*
         * Kernel to scatter
         */
        __global__ void kernScatter(int n, int* odata, const int* scandata, const int* criteria, const int* idata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }
            if (criteria[k] == 1) {
                odata[scandata[k]] = idata[k];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            
            int* dev_data;

            // Extend buffers to handle arrays with lengths which are not a power of two
            int maxDepth = ilog2ceil(n);
            int extended_n = pow(2, maxDepth);

            //dim3 gridSize(32, 32);
            //dim3 blockSize(32, 32);

            dim3 blocksPerGrid((extended_n + blockSize - 1) / blockSize);

            // Memory allocation
            cudaMalloc((void**)&dev_data, sizeof(int) * extended_n);
            checkCUDAError("cudaMalloc dev_data failed!");
            cudaMemset(dev_data, 0, sizeof(int) * extended_n);
            checkCUDAError("cudaMemset dev_data initialization failed!");
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy into dev_data failed!");

            //timer().startGpuTimer();

            // Upsweep - parallel reduction
            for (int d = 0; d < maxDepth; d++) {    // where d is depth of iteration
                int offsetd1 = pow(2, d + 1);
                int offsetd = pow(2, d);
                //int offsetd = pow(2, maxDepth - d - 1);
                kernUpSweepReduction << <blocksPerGrid, blockSize >> > (extended_n, d, offsetd, offsetd1, dev_data);
                checkCUDAError("kernUpStreamReduction invocation failed!");
            }

            // Set last element to identity value which is zero
            cudaMemset(dev_data + extended_n - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset last value to identity failed!");

            // Downsweep
            for (int d = maxDepth - 1; d >= 0; d--) {    // where d is depth of iteration
                int offsetd1 = pow(2, d + 1);
                int offsetd = pow(2, d);
                kernDownSweep << <blocksPerGrid, blockSize >> > (extended_n, d, offsetd, offsetd1, dev_data);
                checkCUDAError("kernDownStream invocation failed!");
            }
            //timer().endGpuTimer();

            //// Getting parallel reduction sum which can be used to convert to inclusive scan
            //int* lastVal = new int();
            //cudaMemcpy(lastVal, dev_data + extended_n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            //checkCUDAError("lastVal memcpy failed!");

            // Copy calculated buffer to output
            cudaMemcpy(odata, dev_data, sizeof(int) * (extended_n), cudaMemcpyDeviceToHost);
            checkCUDAError("odata memcpy failed!");

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
        int compact(int n, int* odata, const int* idata) {
            
            int* dev_idata;
            int* dev_odata;
            int* dev_criteria_buffer;
            int* dev_scanned_buffer;

            int maxDepth = ilog2ceil(n);
            int extended_n = pow(2, maxDepth);

            int* criteria_buffer = new int[extended_n];
            int* scanned_buffer = new int[extended_n];

            dim3 blocksPerGrid((extended_n + blockSize - 1) / blockSize);

            // Memory allocation
            cudaMalloc((void**)&dev_idata, sizeof(int) * extended_n);
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_criteria_buffer, sizeof(int) * extended_n);
            checkCUDAError("cudaMalloc dev_criteria_buffer failed!");
            cudaMalloc((void**)&dev_scanned_buffer, sizeof(int) * extended_n);
            checkCUDAError("cudaMalloc dev_scanned_buffer failed!");

            cudaMemset(dev_idata, 0, sizeof(int) * extended_n);
            checkCUDAError("cudaMemset dev_idata initialization failed!");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy into dev_idata failed!");

            timer().startGpuTimer();
            // Mapping as per criteria
            kernMap << <blocksPerGrid, blockSize >> > (extended_n, dev_criteria_buffer, dev_idata);
            checkCUDAError("kernMap invocation failed!");

            cudaMemcpy(criteria_buffer, dev_criteria_buffer, sizeof(int) * extended_n, cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy into criteria_buffer failed!");

            // Scann criteria buffer to generate scanned buffer
            scan(extended_n, scanned_buffer, criteria_buffer);
            cudaMemcpy(dev_scanned_buffer, scanned_buffer, sizeof(int) * extended_n, cudaMemcpyHostToDevice);
            checkCUDAError("memcpy into dev_scanned_buffer failed!");

            // Malloc for compressed output data, compressed buffer
            // size given by last element of scanned criteria
            cudaMalloc((void**)&dev_odata, sizeof(int) * scanned_buffer[extended_n -1]);
            checkCUDAError("cudaMalloc dev_odata failed!");

            // Initialize odata to 0
            cudaMemset(dev_odata, 0, sizeof(int) * scanned_buffer[extended_n -1]);
            checkCUDAError("cudaMemset dev_odata initialization failed!");

            // Scatter data - insert input data at index obtained
            // from scanned buffer if criteria is set to true
            kernScatter << <blocksPerGrid, blockSize >> > (n, dev_odata, dev_scanned_buffer, dev_criteria_buffer, dev_idata);
            checkCUDAError("kernMap invocation failed!");

            timer().endGpuTimer();

            // Copy calculated buffer to output
            cudaMemcpy(odata, dev_odata, sizeof(int) * scanned_buffer[extended_n -1], cudaMemcpyDeviceToHost);
            checkCUDAError("odata memcpy failed!");

            cudaFree(dev_scanned_buffer);
            cudaFree(dev_criteria_buffer);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            /*delete[] criteria_buffer;
            delete[] scanned_buffer;*/

            return scanned_buffer[extended_n-1];
        }
    }
}
