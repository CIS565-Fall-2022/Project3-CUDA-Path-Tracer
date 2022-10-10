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
         * (Optional) For better understanding before starting moving to GPU,
         you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            odata[0] = 0;
            for (int i = 1; i < n; i++)
            {
                odata[i] = odata[i-1] + idata[i-1];
            }
           
            //Why the last two digit different?
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        //Well I don't know exactly the condition
        //So I treat it as remove 0 I guess
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int j = 0;
            for (int i = 0; i < n; i++)
            {
                if (idata[i] > 0)
                {
                    odata[j] = idata[i];
                    j++;
                }
            }
            timer().endCpuTimer();
            return j;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* boolArray = new int[n * sizeof(int)];
            int* scanArray = new int[n * sizeof(int)];
            timer().startCpuTimer();
            // TODO
            for (int i = 0; i < n; i++)
            {
                boolArray[i] = (idata[i] > 0) ? 1 : 0;
            }
            //Set temp array

            //begin scan
            //Inclusive scan
            scanArray[0] = boolArray[0]; //identity
            for (int i = 1; i < n; i++)
            {
                scanArray[i] = scanArray[i-1] + boolArray[i];
            }
            int elementNum = scanArray[n - 1];
            //Shift to right
            //Exclusive scan
            for (int i = n; i > 0; i--)
            {
                scanArray[i] = scanArray[i - 1];
            }
            scanArray[0] = 0;
            //Scatter
            for (int i = 0; i < n; i++)
            {
                if (boolArray[i] > 0)
                {
                    odata[scanArray[i]]=idata[i];
                }
            }
            timer().endCpuTimer();

            return elementNum;
        }
    }
}
