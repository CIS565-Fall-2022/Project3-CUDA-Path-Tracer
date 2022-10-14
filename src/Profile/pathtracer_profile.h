#pragma once
#include <unordered_map>
#include <string>
#include "timer.h"
#include "../utilities.h"
#include "../pathtrace.h"

namespace PathTracer {
	class GPUProfileScope {
		Profiling::Timer t;
		std::string id;
	public:
		GPUProfileScope(std::string const& id) : id(id) {
			t.startGpuTimer();
		}
		~GPUProfileScope() {
			t.endGpuTimer();
			GetProfileData()[id].add_time(t.getGpuElapsedTimeForPreviousOperation());
		}
		GPUProfileScope(GPUProfileScope const&) = delete;
		GPUProfileScope(GPUProfileScope&&) = delete;
	};

	class ProfileHelper {
		float total_t;
		Profiling::Timer t;
		std::string id;

	public:
		ProfileHelper(std::string const& id) : total_t(0), id(id) { }
		~ProfileHelper() {
			GetProfileData()[id].add_time(total_t);
		}

		// not using forwarding refs here because params must be passed by copy to GPU
		template<typename T, typename... Args>
		void call(T kern, dim3 x, dim3 y, Args... args) {
#ifdef PROFILE
			t.startGpuTimer();
			
			kern KERN_PARAM(x, y) (args...);
			cudaDeviceSynchronize();

			t.endGpuTimer();
			total_t += t.getGpuElapsedTimeForPreviousOperation();
#else
			kern KERN_PARAM(x, y) (args...);
#endif // PROFILE
		}
		void begin() {
#ifdef PROFILE
			t.startGpuTimer();
#endif
		}
		void end() {
#ifdef PROFILE
			t.endGpuTimer();
			total_t += t.getGpuElapsedTimeForPreviousOperation();
#endif
		}
		float total_time() const {
			return total_t;
		}

		ProfileHelper(ProfileHelper const&) = delete;
		ProfileHelper(ProfileHelper&&) = delete;
	};
}