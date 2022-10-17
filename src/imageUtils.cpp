#include "imageUtils.h"
#include "image.h"
#include "main.h"

#include <sstream>
#include <algorithm>
#include <numeric>

void ImageUtils::SaveImage(glm::vec3 const* src, std::string const& name, bool radiance) {
    float samples = g_iteration;
    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = x + (y * width);
            glm::vec3 pix = src[index];
            if (radiance) pix /= samples;
            img.setPixel(width - 1 - x, y, pix);
        }
    }

    std::ostringstream ss;
    ss << name << "." << currentTimeString() << "." << samples << "samp";

    img.savePNG(ss.str());
}

void ImageUtils::SaveImage(RenderState const* state) {
    SaveImage(state->image.data(), state->imageName, true);
}

void ImageUtils::SaveImage(uchar4 const* pbo) {
    int sz = width * height;
    uchar4* img = new uchar4[sz];
    glm::vec3* img_normalized = new glm::vec3[sz];
    D2H(img, pbo, size_t(width) * height);
    std::transform(img, img + sz, img_normalized, [](uchar4 const& pix) {
        return glm::vec3(pix.x / 255.f, pix.y / 255.f, pix.z / 255.f);
    });
    SaveImage(img_normalized, "pbo", false);
    delete[] img;
    delete[] img_normalized;
}

struct MSE_OP {
    float operator()(glm::vec3 const& lhs, glm::vec3 const& rhs) const {
        return glm::length2(lhs - rhs);
    }
};
float ImageUtils::CalculateMSE(int size, glm::vec3 const* img1, glm::vec3 const* img2) {
    return (1.f / size) * std::inner_product(img1, img1 + size, img2, 0, std::plus<float>(), MSE_OP());
}
float ImageUtils::CalculatePSNR(int size, glm::vec3 const* img1, glm::vec3 const* img2) {
    float mse = CalculateMSE(size, img1, img2);
    if (glm::abs(mse) <= EPSILON) {
        return 100.f; // no noise
    }
    return 20.f * (glm::log(255.f / glm::sqrt(mse)) / glm::log(10));
}