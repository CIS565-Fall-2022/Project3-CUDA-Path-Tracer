#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            thrust::host_vector<int> host_in(n);
            thrust::host_vector<int> host_out(n);

            // Copy input into host vectors
            for (int i = 0; i < n; ++i) {
                host_in[i] = idata[i];
            }
            
            // Create device_vectors from host_vectors
            thrust::device_vector<int> dev_in(host_in);
            thrust::device_vector<int> dev_out(host_out);

            timer().startGpuTimer();
            thrust::exclusive_scan(dev_in.begin(), dev_in.end(), dev_out.begin());
            timer().endGpuTimer();

            // Write final results
            for (int i = 0; i < n; i++) {
                odata[i] = dev_out[i];
            }
        }
    }
}
