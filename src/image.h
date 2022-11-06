#pragma once
#include <glm/glm.hpp>

struct RenderState;
struct uchar4;

class Image {
private:
    int xSize;
    int ySize;
    glm::vec3 *pixels;

public:
    Image(int x, int y);
    Image(std::string const& file);
    Image(int x, int y, uchar4 const* dev_img);
    ~Image();

    glm::vec3 const* getPixels() const { return pixels; }
    void setPixel(int x, int y, const glm::vec3 &pixel);
    void savePNG(const std::string &baseFilename) const;
    void saveHDR(const std::string &baseFilename) const;
};