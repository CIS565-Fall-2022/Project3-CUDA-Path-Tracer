#pragma once

#include "glm/glm.hpp"
#include "intellisense.h"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;
};

// convenience macro
#ifndef NDEBUG
void checkCUDAErrorFn(const char* msg, const char* file, int line);
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
#define CHECK_CUDA(func_call) do { if(func_call != cudaSuccess) checkCUDAError("cuda func failed:\n" ## #func_call); } while(0)
#define PRINT_GPU(dev_arr, size) printGPU(#dev_arr, dev_arr, size)
#else
#define checkCUDAError(msg) (void)0
#define CHECK_CUDA(func_call) (void)0
#define PRINT_GPU(dev_arr, size) (void)0
#endif // !NDEBUG

#define ALLOC(name, size) CHECK_CUDA(cudaMalloc((void**)&(name), (size) * sizeof(*name)))
#define MEMSET(name, val, size) CHECK_CUDA(cudaMemset(name, val, size))
#define FREE(name) CHECK_CUDA(cudaFree(name))
#define H2D(dev_name, name, size) CHECK_CUDA(cudaMemcpy(dev_name, name, (size) * sizeof(*name), cudaMemcpyHostToDevice))
#define D2H(name, dev_name, size) CHECK_CUDA(cudaMemcpy(name, dev_name, (size) * sizeof(*name), cudaMemcpyDeviceToHost))
#define D2D(dev_name1, dev_name2, size) CHECK_CUDA(cudaMemcpy(dev_name1, dev_name2, (size) * sizeof(*dev_name1), cudaMemcpyDeviceToDevice))

template<typename T>
static inline void printGPU(char const* name, T * dev, int n) {
    T* tmp = new T[n];
    std::cout << name << "\n";
    D2H(tmp, dev, n);
    for (int i = 0; i < n; ++i)
        std::cout << tmp[i] << " \n"[i < n - 1 ? 0 : 1];
    delete[] tmp;
}

template<typename T>
static inline T getGPU(T * dev, int i) {
    T tmp;
    D2H(&tmp, dev + i, 1);
    return tmp;
}
template<typename T>
static inline void setGPU(T * dev, int i, T val) {
    H2D(dev + i, &val, 1);
}

namespace utilityCore {
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}
