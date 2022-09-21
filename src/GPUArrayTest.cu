#include "GPUArrayTest.h"

namespace GPUArrayTest {
	static __global__ void unit_test_kernel(GPUArray<int> arr) {
		for (int i = 0; i < arr.size(); ++i) {
			printf("%d ", arr[i]);
		}
		printf("\n");
	}
	static __global__ void unit_test_kernel(GPUArray<glm::ivec3> arr) {
		for (int i = 0; i < arr.size(); ++i) {
			printf("[%d,%d,%d] ", arr[i].x, arr[i].y, arr[i].z);
			arr[i].x += 1;
		}
		printf("\n");
		for (int i = 0; i < arr.size(); ++i) {
			printf("[%d,%d,%d] ", arr[i].x, arr[i].y, arr[i].z);
		}
		printf("\n");
	}

	void unit_test() {
#ifndef NDEBUG
		std::cout << "test GPU Array...\n";
		// raw array
		int data[10];
		for (int i = 0; i < 10; ++i) {
			data[i] = rand() % 15;
		}

		int* dev_data; ALLOC(dev_data, 10);
		H2D(dev_data, data, 10);

		GPUArray<int> dev_arr;
		dev_arr.resize(10).copy_from(dev_data);
		FREE(dev_data);

		unit_test_kernel KERN_PARAM(1, 1) (dev_arr);

		glm::ivec3 vec_data[3]{
			{1,0,0},
			{0,1,0},
			{0,0,1}
		};

		GPUArray<glm::ivec3> dev_vec_arr;
		dev_vec_arr.resize(3).copy_from(vec_data);
		unit_test_kernel KERN_PARAM(1, 1) (dev_vec_arr);
#endif
	}
}