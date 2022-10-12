#pragma once
#include "Denoise/denoise.h"

class octree;

class GuiDataContainer
{
public:
    GuiDataContainer();
    ~GuiDataContainer();
    void Reset();

    bool hide_gui;
    int traced_depth;
    char** scene_file_names;
    char** save_file_names;
    int num_saves;
    int num_scenes;
    int cur_scene;
    int cur_save;
    char save_file_name_buf[256];
    char const* prompt_text;
    bool draw_coord_frame;
    bool draw_debug_aabb;
    bool draw_world_aabb;
    bool draw_GPU_tree;
    int octree_depth;
    int octree_depth_filter;
    int octree_intersection_cnt;
    octree* test_tree;

    Denoiser::ParamDesc desc;
    struct {
        bool is_on;    
        int debug_tex_idx;
    } denoiser_options;
};