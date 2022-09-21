#pragma once
#include <cuda_runtime.h>

struct PathSegment;

struct PathSegPool {
	PathSegPool();
	~PathSegPool();

	void update();
	static void unit_test();
	__host__ __device__ size_t size() { return compact_size; }
	__host__ __device__ PathSegment* input_buf() { return dev_bufs[buf_idx]; }
	__host__ __device__ PathSegment* output_buf() { return dev_bufs[1 - buf_idx]; }
	__device__ void dev_set(int idx, PathSegment const* val);
	__device__ PathSegment const* dev_get(int idx);
private:
	size_t compact_size;
	int buf_idx;
	PathSegment* dev_bufs[3];
};