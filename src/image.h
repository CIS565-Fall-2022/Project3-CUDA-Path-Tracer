#pragma once

#include <glm/glm.hpp>
#include <vector_types.h>
struct RenderState;
class image {
private:
    int xSize;
    int ySize;
    glm::vec3 *pixels;

public:
    image(int x, int y);
    ~image();
    void setPixel(int x, int y, const glm::vec3 &pixel);
    void savePNG(const std::string &baseFilename);
    void saveHDR(const std::string &baseFilename);
};

void saveImage(glm::vec3 const* src, std::string const& name, bool radiance);
void saveImage(RenderState const* state);
void saveImage(uchar4 const* pbo);