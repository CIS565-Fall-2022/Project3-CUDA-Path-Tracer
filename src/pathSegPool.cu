#include "pathSegPool.h"
#include "sceneStructs.h"
#include "utilities.h"
#include "efficient.h"
#include "GPUArray.h"

static constexpr int OUT_BUF_IDX = 2;
static constexpr size_t MAX_PATH_SEGS = 1 << 12;

PathSegPool::PathSegPool() : buf_idx(0), compact_size(0) {
	for (int i = 0; i < 3; ++i) {
		ALLOC(dev_bufs[i], MAX_PATH_SEGS);
		MEMSET(dev_bufs[i], 0, MAX_PATH_SEGS * sizeof(PathSegment));
	}
}
PathSegPool::~PathSegPool() {
	for (int i = 0; i < 3; ++i) {
		FREE(dev_bufs[i]);
	}
}

__device__ void 
PathSegPool::dev_set(int idx, PathSegment const* val) {
	output_buf()[idx] = *val;
}
__device__ PathSegment const*
PathSegPool::dev_get(int idx) {
	return input_buf() + idx;
}

void PathSegPool::update() {
	// compact current buffer
	compact_size = StreamCompaction::compact(MAX_PATH_SEGS, dev_bufs[buf_idx], dev_bufs[buf_idx]);
	// swap buffers
	buf_idx = 1 - buf_idx;
	// zero the current buffer
	MEMSET(dev_bufs[buf_idx], 0, MAX_PATH_SEGS * sizeof(PathSegment));
}

void PathSegPool::unit_test() {
#ifndef NDEBUG
	char const* line_sep = "=====================";
#define SECTION(str) std::cout << line_sep << " " << str << " " << line_sep << std::endl;
	SECTION("test compaction, primitive")
	{
		int tmp_in[]{ 1,0,2,0,33,0,0,78,12,0,11,19,0,0,17,0 };
		for (int x : tmp_in) {
			std::cout << x << " ";
		}
		std::cout << std::endl;

		int* dev_tmp_in;
		int* dev_tmp_out;
		ALLOC(dev_tmp_in, 16); H2D(dev_tmp_in, tmp_in, 16);
		ALLOC(dev_tmp_out, 16);
		std::cout << "compaction result: \n";
		PRINT_GPU(dev_tmp_out, StreamCompaction::compact(16, dev_tmp_out, dev_tmp_in));
		std::cout << "compaction in place result: \n";
		PRINT_GPU(dev_tmp_in, StreamCompaction::compact(16, dev_tmp_in, dev_tmp_in));
		FREE(dev_tmp_in);
		FREE(dev_tmp_out);
	}
	SECTION("test compaction, GPU Array, primitive")
	{
		int tmp_in[]{ 1,0,2,0,33,0,0,78,12,0,11,19,0,0,17,0 };
		for (int x : tmp_in) {
			std::cout << x << " ";
		}
		std::cout << std::endl;

		GPUArray<int> dev_tmp_in(16);
		GPUArray<int> dev_tmp_out(16);
		dev_tmp_in.copy_from(tmp_in);
		std::cout << "compaction result: \n";
		int* raw_out = dev_tmp_out;
		int* raw_in = dev_tmp_in;

		PRINT_GPU(raw_out, StreamCompaction::compact(16, raw_out, raw_in));
		std::cout << "compaction in place result: \n";
		PRINT_GPU(raw_in, StreamCompaction::compact(16, raw_in, raw_in));
	}
	SECTION("test compaction, struct")
	{
		PathSegment tmp_in[10];
		for (int i = 0; i < 10; ++i) {
			auto& x = tmp_in[i];
			if (i & 1) {
				x.terminate();
			}

			printf("{%f,%f,%f,%d,%f,%f,%f,%d}\n", 
				x.color.x, x.color.y, x.color.z,
				x.pixelIndex,
				x.ray.direction.x,
				x.ray.origin.y,
				x.remainingBounces);
		}

		GPUArray<PathSegment> dev_arr(10);
		dev_arr.copy_from(tmp_in);
		std::cout << "compaction in place result: \n";
		thrust::device_ptr<PathSegment> raw_arr((PathSegment*)dev_arr);
		int n = StreamCompaction::compact(10, raw_arr, raw_arr);
		PathSegment* tmp = new PathSegment[n];
		D2H(tmp, dev_arr, n);

		for (int i = 0; i < n; ++i) {
			auto const& x = tmp[i];
			printf("{%f,%f,%f,%d,%f,%f,%f,%d}\n",
				x.color.x, x.color.y, x.color.z,
				x.pixelIndex,
				x.ray.direction.x,
				x.ray.origin.y,
				x.remainingBounces);
		}
		delete[] tmp;
	}
	checkCUDAError("unit test");

#endif // !NDEBUG
}