#include "timer.h"
#include <cuda_runtime.h>
#include <stdexcept>


Profiling::Timer::Timer() {
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_end);
}

Profiling::Timer::~Timer() {
    cudaEventDestroy(event_start);
    cudaEventDestroy(event_end);
}
void Profiling::Timer::startCpuTimer() {
    if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
    cpu_timer_started = true;

    time_start_cpu = std::chrono::high_resolution_clock::now();
}
void Profiling::Timer::endCpuTimer()
{
    time_end_cpu = std::chrono::high_resolution_clock::now();

    if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }

    std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
    prev_elapsed_time_cpu_milliseconds =
        static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

    cpu_timer_started = false;
}

void Profiling::Timer::startGpuTimer()
{
    if (gpu_timer_started) { throw std::runtime_error("GPU timer already started"); }
    gpu_timer_started = true;

    cudaEventRecord(event_start);
}
void Profiling::Timer::endGpuTimer()
{
    cudaEventRecord(event_end);
    cudaEventSynchronize(event_end);

    if (!gpu_timer_started) { throw std::runtime_error("GPU timer not started"); }

    cudaEventElapsedTime(&prev_elapsed_time_gpu_milliseconds, event_start, event_end);
    gpu_timer_started = false;
}
float Profiling::Timer::getCpuElapsedTimeForPreviousOperation() noexcept
{
    return prev_elapsed_time_cpu_milliseconds;
}

float Profiling::Timer::getGpuElapsedTimeForPreviousOperation() noexcept
{
    return prev_elapsed_time_gpu_milliseconds;
}