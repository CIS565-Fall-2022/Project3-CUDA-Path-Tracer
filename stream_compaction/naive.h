#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

        __global__ void kernScan(int n, int d, int* odata, int* idata);

        void scan(int n, int *odata, const int *idata);

        void scan2(int n, int* odata, const int* idata);
    }
}
