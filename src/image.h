#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

class Image {
public:
    Image(int width, int height);
    Image(const std::string& filename);
    ~Image();

    void setPixel(int x, int y, const glm::vec3& pixel);
    void savePNG(const std::string& baseFilename);
    void saveHDR(const std::string& baseFilename);

    size_t size() const {
        return sizeof(glm::vec3) * mWidth * mHeight;
    }

private:
    int mWidth;
    int mHeight;
    glm::vec3* mPixels = nullptr;
};

struct DevTexture {
    int width;
    int height;
    glm::vec3* data;
};

__device__ inline glm::vec3 texelFetch(const DevTexture& tex, int x, int y) {
    return tex.data[y * tex.width + x];
}

//__device__ inline glm::vec3 texelFetch(const glm::vec3* data, const DevTexture& tex, int x, int y) {
//}

//__device__ static glm::vec3 texture2DLinear(const DevTexture& tex, glm::vec2 uv) {
//    const float Eps = 1e-6f;
//
//    uv = glm::fract(uv);
//    float fx = uv.x * (tex.width - Eps);
//    float fy = uv.y * (tex.height - Eps);
//
//
//    int ix = glm::fract(fx) < .5f ? int(fx) : int(fx) - 1;
//    if (ix < 0) {
//        ix += tex.width;
//    }
//    int iy = glm::fract(fy) < .5f ? int(fy) : int(fy) - 1;
//    if (iy < 0) {
//        iy += tex.height;
//    }
//
//    int ixp = ix + 1;
//    if (ixp >= tex.width) {
//        ixp -= tex.width;
//    }
//    int iyp = iy + 1;
//    if (iyp >= tex.height) {
//        iyp -= tex.height;
//    }
//
//    glm::vec3 c1 = texelFetch(tex, ix, iy);
//    glm::vec3 c2 = texelFetch(tex, ixp, iy);
//    glm::vec3 c3 = texelFetch(tex, ix, iyp);
//    glm::vec3 c4 = texelFetch(tex, ixp, iyp);
//
//    return glm::mix(
//        glm::mix(c1, c2, 
//}