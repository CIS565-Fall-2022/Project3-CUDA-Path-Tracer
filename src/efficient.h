#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace StreamCompaction {
    template<typename T>
    struct Pred {
        __device__ bool operator()(T const& x) const {
            return !x;
        }
    };

    /**
     * Performs prefix-sum (aka scan) on idata, storing the result into odata.
     */
    template<typename T>
    void scan(int n, T* out, const T* in) {
        thrust::device_vector<T> dev_in(in, in + n);
        thrust::device_vector<T> dev_out(n);
        thrust::exclusive_scan(dev_in.begin(), dev_in.end(), dev_out.begin());
        thrust::copy(dev_out.begin(), dev_out.end(), out);
    }
    template<typename T>
    int compact(int n, T* out, const T* in) {
        thrust::device_vector<T> dev_in(in, in + n);
        auto it = thrust::remove_if(dev_in.begin(), dev_in.end(), Pred<T>());
        int ret = thrust::distance(dev_in.begin(), it);
        thrust::copy(dev_in.begin(), it, out);
        return ret;
    }
    template<typename T>
    void sort(int n, T* out, const T* in) {
        thrust::device_vector<T> dev_in(in, in + n);
        thrust::sort(dev_in.begin(), dev_in.end());
        thrust::copy(dev_in.begin(), dev_in.end(), out);
    }
};