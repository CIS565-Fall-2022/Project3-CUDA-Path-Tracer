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
            //timer().startCpuTimer();
            int identity = 0;

            odata[0] = identity;
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];     // exclusive scan
            }

            //timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int nonZeroIdx = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[nonZeroIdx] = idata[i];
                    nonZeroIdx++;
                }
            }

            timer().endCpuTimer();
            return nonZeroIdx;
        }

        void map(int n, int* odata, const int* idata) {
            for (int i = 0; i < n; i++) {
                odata[i] = (idata[i] == 0) ? 0 : 1;
            }
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* mapped = new int[n];
            int* scanned = new int[n];
            map(n, mapped, idata);
            scan(n, scanned, mapped);
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (mapped[i] == 1) {
                    int index = scanned[i];
                    odata[index] = idata[i];
                    count++;
                }
            }
            delete[] mapped;
            delete[] scanned;
            timer().endCpuTimer();
            return count;
        }
    }
}
