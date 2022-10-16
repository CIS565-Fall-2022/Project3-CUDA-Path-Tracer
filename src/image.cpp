#include <iostream>
#include <string>
#include <stb_image_write.h>
#include "image.h"
#include "main.h"

image::image(int x, int y) :
        xSize(x),
        ySize(y),
        pixels(new glm::vec3[x * y]) {
}

image::~image() {
    delete pixels;
}

void image::setPixel(int x, int y, const glm::vec3 &pixel) {
    assert(x >= 0 && y >= 0 && x < xSize && y < ySize);
    pixels[(y * xSize) + x] = pixel;
}

void image::savePNG(const std::string &baseFilename) {
    unsigned char *bytes = new unsigned char[3 * xSize * ySize];
    for (int y = 0; y < ySize; y++) {
        for (int x = 0; x < xSize; x++) { 
            int i = y * xSize + x;
            glm::vec3 pix = glm::clamp(pixels[i], glm::vec3(), glm::vec3(1)) * 255.f;
            bytes[3 * i + 0] = (unsigned char) pix.x;
            bytes[3 * i + 1] = (unsigned char) pix.y;
            bytes[3 * i + 2] = (unsigned char) pix.z;
        }
    }

    std::string filename = baseFilename + ".png";
    stbi_write_png(filename.c_str(), xSize, ySize, 3, bytes, xSize * 3);
    std::cout << "Saved " << filename << "." << std::endl;

    delete[] bytes;
}

void image::saveHDR(const std::string &baseFilename) {
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), xSize, ySize, 3, (const float *) pixels);
    std::cout << "Saved " + filename + "." << std::endl;
}

void saveImage(glm::vec3 const* src, std::string const& name, bool radiance) {
    float samples = g_iteration;
    // output image file
    image img(width, height);

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

void saveImage(RenderState const* state) {
    saveImage(state->image.data(), state->imageName, true);
}

void saveImage(uchar4 const* pbo) {
    int sz = width * height;
    uchar4* img = new uchar4[sz];
    glm::vec3* img_normalized = new glm::vec3[sz];
    D2H(img, pbo, width * height);
    std::transform(img, img + sz, img_normalized, [](uchar4 const& pix) {
        return glm::vec3(pix.x / 255.f, pix.y / 255.f, pix.z / 255.f);
    });
    saveImage(img_normalized, "pbo", false);
    delete[] img;
    delete[] img_normalized;
}