#pragma once
#include "utilities.h"
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <exception>

template<typename T>
class GPUArray {
	T* _raw_ptr;
	size_t _size;
	bool _is_copy;

public:
	GPUArray() : _raw_ptr(nullptr), _size(0), _is_copy(false) { }
	explicit GPUArray(size_t size) : _raw_ptr(nullptr), _size(size), _is_copy(false) {
		ALLOC(_raw_ptr, size);
		MEMSET(_raw_ptr, 0, sizeof(T) * size);
	}
	GPUArray(GPUArray const& o) : _raw_ptr(o._raw_ptr), _size(o._size), _is_copy(true) {}

	GPUArray& resize(size_t new_size) {
		if (_is_copy) {
			throw std::runtime_error("cannot modify a copy of GPU Array");
		}

		if (_size == new_size) {
			return *this;
		}

		T* new_ptr;
		ALLOC(new_ptr, new_size);

		if (_raw_ptr) {
			thrust::device_ptr<T> dev_raw(_raw_ptr);
			thrust::device_ptr<T> dev_new(new_ptr);

			if (_size < new_size) {
				thrust::copy_n(dev_raw, _size, dev_new);
				MEMSET(new_ptr + _size, 0, sizeof(T) * (new_size - _size));
			} else { // if(size > new_size)
				thrust::copy_n(dev_raw, new_size, dev_new);
			}
			FREE(_raw_ptr);
		}
		_raw_ptr = new_ptr;
		_size = new_size;
		return *this;
	}

	template<typename SRC_T>
	GPUArray& copy_from(SRC_T src) {
		if (_is_copy) {
			throw std::runtime_error("cannot modify a copy of GPU Array");
		}
		thrust::copy_n(src, _size, thrust::device_ptr<T>(_raw_ptr));
		return *this;
	}

	GPUArray& zero_mem() {
		if (_raw_ptr) {
			MEMSET(_raw_ptr, 0, sizeof(T) * _size);
		}
		return *this;
	}

	~GPUArray() {
		if (!_is_copy && _raw_ptr) {
#ifndef NDEBUG
			std::cout << "free\n";
#endif
			FREE(_raw_ptr);
		}
	}
	__host__ __device__ size_t size() {
		return _size;
	}
	__host__ __device__ operator T* () {
		return _raw_ptr;
	}
	operator thrust::device_ptr<T>() {
		return thrust::device_ptr<T>(_raw_ptr);
	}
	__device__ T& operator[](int idx) {
#ifndef NDEBUG
		if (idx < 0 || idx >= _size) {
			assert(!"out of bound access");
		}
#endif // !NDEBUG
		return _raw_ptr[idx];
	}

};