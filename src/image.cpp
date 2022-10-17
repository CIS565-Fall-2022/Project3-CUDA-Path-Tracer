#include <iostream>
#include <glm/glm.hpp>
#include <stb_image.h>
#include <stb_image_write.h>
#include <vector_types.h>
#include "utilities.h"
#include "image.h"
#include "main.h"

Image::Image(int x, int y) :
        xSize(x),
        ySize(y),
        pixels(new glm::vec3[size_t(x) * y]) { }

Image::Image(std::string const& file) {
    int comp;
    float* data = stbi_loadf(file.c_str(), &xSize, &ySize, &comp, 3);
    pixels = new glm::vec3[size_t(xSize) * ySize];

    for (size_t i = 2; i < size_t(xSize) * ySize * 3; i += 3) {
        pixels[i / 3] = glm::vec3(data[i - 2], data[i - 1], data[i]);
    }
    stbi_image_free(data);
}

Image::Image(int x, int y, uchar4 const* dev_img) :
    xSize(x),
    ySize(y),
    pixels(new glm::vec3[size_t(x) * y])
{
    uchar4* tmp = new uchar4[size_t(x) * y];
    D2H(tmp, dev_img, size_t(x) * y);
    for (size_t i = 0; i < size_t(x) * y; ++i) {
        pixels[i] = glm::vec3(tmp[i].x / 255.f, tmp[i].y / 255.f, tmp[i].z / 255.f);
    }

    delete[] tmp;
}

Image::~Image() {
    delete[] pixels;
}

void Image::setPixel(int x, int y, const glm::vec3 &pixel) {
    assert(x >= 0 && y >= 0 && x < xSize && y < ySize);
    pixels[(y * xSize) + x] = pixel;
}

void Image::savePNG(const std::string &baseFilename) {
    unsigned char *bytes = new unsigned char[3 * size_t(xSize) * ySize];
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

void Image::saveHDR(const std::string &baseFilename) {
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), xSize, ySize, 3, (const float *) pixels);
    std::cout << "Saved " + filename + "." << std::endl;
}