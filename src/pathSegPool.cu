#include "pathSegPool.h"
#include "sceneStructs.h"
#include "utilities.h"
#include "efficient.h"

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
	std::cout << "test compaction ... \n";
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
#endif // !NDEBUG
}