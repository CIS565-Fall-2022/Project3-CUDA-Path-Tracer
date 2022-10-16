#pragma once
#include "Denoise/denoise.h"
#include <vector>
#include <string>

class octree;
class GuiDataContainer
{
public:
    static constexpr int BUF_SIZE = 256;
private:
    struct TextBuf {
        char* data;
        TextBuf() { data = new char[BUF_SIZE]; memset(data, 0, sizeof(*data) * BUF_SIZE); }
        ~TextBuf() { delete[] data; }
    };
public:
    GuiDataContainer();
    ~GuiDataContainer();
    void Reset();
    void ClearBuf();
    char* NextBuf();

    bool hide_gui;
    int traced_depth;
    bool draw_coord_frame;
    bool draw_debug_aabb;
    bool draw_world_aabb;
    bool draw_GPU_tree;
    int octree_depth;
    int octree_depth_filter;
    int octree_intersection_cnt;
    octree* test_tree;
    Denoiser::ParamDesc desc;

    std::string cur_scene;
    std::vector<TextBuf> char_bufs;

    struct {
        bool is_on;    
        int debug_tex_idx;
    } denoiser_options;
};