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

    int width() const {
        return mWidth;
    }

    int height() const {
        return mHeight;
    }

    size_t byteSize() const {
        return sizeof(glm::vec3) * mWidth * mHeight;
    }

    glm::vec3* data() const {
        return mPixels;
    }

private:
    int mWidth;
    int mHeight;
    glm::vec3* mPixels = nullptr;
};

struct DevTextureObj {
    DevTextureObj() = default;

    DevTextureObj(Image* img, glm::vec3 *devData) :
        width(img->width()), height(img->height()), devData(devData) {}

    __device__ glm::vec3 fetchTexel(int x, int y) {
        return devData[y * width + x];
    }

    __device__ glm::vec3 linearSample(glm::vec2 uv) {
        const float Eps = FLT_MIN * 2.f;
        uv = glm::fract(uv);

        float fx = uv.x * (width - Eps);
        float fy = uv.y * (height - Eps);

        int ix = glm::fract(fx) < .5f ? int(fx) : int(fx) - 1;
        if (ix < 0) {
            ix += width;
        }
        int iy = glm::fract(fy) < .5f ? int(fy) : int(fy) - 1;
        if (iy < 0) {
            iy += height;
        }

        int ux = ix + 1;
        if (ux >= width) {
            ux -= width;
        }
        int uy = iy + 1;
        if (uy >= width) {
            uy -= height;
        }

        float lx = glm::fract(fx + .5f);
        float ly = glm::fract(fy + .5f);

        // Bilinear interpolation
        return glm::mix(
            glm::mix(fetchTexel(ix, iy), fetchTexel(ux, iy), lx),
            glm::mix(fetchTexel(ix, uy), fetchTexel(ux, uy), lx),
            ly
        );
    }

    int width;
    int height;
    glm::vec3* devData;
};