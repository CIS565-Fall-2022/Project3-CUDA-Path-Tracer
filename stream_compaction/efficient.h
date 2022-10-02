#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void doScan(int n, int* dev_odata);

        void scan(int n, int* odata, const int* idata);

        int compact(int n, int *odata, const int *idata);
    }
}
