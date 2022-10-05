#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
              odata[i] = odata[i - 1] + idata[i - 1];
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
          /* Start Timer */
            timer().startCpuTimer();
            // TODO
            int currIndex = 0;
            for (int i = 0; i < n; i++) {
              if (idata[i] != 0) {
                odata[currIndex++] = idata[i];
              }
            }
            timer().endCpuTimer();
            return currIndex;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            // TODO
            int* tdata = new int[n];

            timer().startCpuTimer();

            for (int k = 0; k < n; k++)
              tdata[k] = (idata[k] == 0) ? 0: 1;

            odata[0] = 0;
            for (int i = 1; i < n; i++) {
              odata[i] = odata[i - 1] + tdata[i - 1];
            }

            for (int k = 0; k < n; k++) {
              if (tdata[k] == 1) {
                odata[odata[k]] = idata[k];
              }
            }

            timer().endCpuTimer();

            delete[] tdata;

            return odata[n - 1];
        }
    }
}
