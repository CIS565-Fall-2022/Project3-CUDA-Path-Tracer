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
         
            doScan(n, odata, idata);

            timer().endCpuTimer();
        }

        void doScan(int n, int* odata, const int* idata)
        {
            // Exclusive scan
            odata[0] = 0;
            for (int i = 1; i < n; ++i)
            {
                odata[i] = idata[i - 1] + odata[i - 1];
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO

            int count = 0;
            for (int i = 0; i < n; ++i)
            {
                if (idata[i] > 0)
                {
                    odata[count++] = idata[i];
                }
            }

            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {

            int* tmpInputData = new int[n];
            int* tmpOutputData = new int[n];

            timer().startCpuTimer();

            // Transfer idata to 0,1 set
            for (int i = 0; i < n; ++i)
            {
                tmpInputData[i] = idata[i] > 0 ? 1 : 0;
            }

            // Exclusive scan
            doScan(n, tmpOutputData, tmpInputData);

            // Final array size
            int count = tmpOutputData[n - 1] + tmpInputData[n - 1];

            // Scatter
            for (int i = 0; i < n; ++i)
            {
                if (tmpInputData[i] > 0)
                {
                    odata[tmpOutputData[i]] = idata[i];
                }
            }

            timer().endCpuTimer();

            delete[] tmpInputData;
            delete[] tmpOutputData;
          
            return count;
        }
    }
}
