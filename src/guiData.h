#pragma once

class octree;
class GuiDataContainer
{
public:
    GuiDataContainer();
    ~GuiDataContainer();
    void Reset();
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

    struct {
        bool is_on;
        int filter_size;
        float color_weight;
        float normal_weight;
        float pos_weight;
        bool show_gbuf;
    } denoiser_options;
};