#pragma once
#include <driver_types.h>
#include <chrono>
#include <sstream>

namespace Profiling {
    class ProfileData {
        float cur_time;
        float tot_time;
        size_t num_times;
    public:
        ProfileData() : cur_time(0), tot_time(0), num_times(0) {}
        void add_time(float t) {
            cur_time = t;
            tot_time += t;
            ++num_times;
        }
        float get_cur_time() const { return cur_time; }
        float get_ave_time() const { return num_times ? tot_time / num_times : 0; }
        std::string to_string() const {
            std::ostringstream oss;
            oss << "cur = " << get_cur_time() << "ms\nave = " << get_ave_time() << "ms";
            return oss.str();
        }
        void clear() {
            cur_time = tot_time = 0.f;
            num_times = 0;
        }
    };

/**
* This class is used for timing the performance
* Uncopyable and unmovable
*
* Adapted from WindyDarian(https://github.com/WindyDarian)
*/
    class Timer
    {
    public:
        Timer();
        ~Timer();
        void startCpuTimer();
        void endCpuTimer();
        void startGpuTimer();
        void endGpuTimer();
        float getCpuElapsedTimeForPreviousOperation() noexcept;
        float getGpuElapsedTimeForPreviousOperation() noexcept;

        // remove copy and move functions
        Timer(const Timer&) = delete;
        Timer(Timer&&) = delete;
        Timer& operator=(const Timer&) = delete;
        Timer& operator=(Timer&&) = delete;

    private:
        cudaEvent_t event_start = nullptr;
        cudaEvent_t event_end = nullptr;

        using time_point_t = std::chrono::high_resolution_clock::time_point;
        time_point_t time_start_cpu;
        time_point_t time_end_cpu;

        bool cpu_timer_started = false;
        bool gpu_timer_started = false;

        float prev_elapsed_time_cpu_milliseconds = 0.f;
        float prev_elapsed_time_gpu_milliseconds = 0.f;
    };
}