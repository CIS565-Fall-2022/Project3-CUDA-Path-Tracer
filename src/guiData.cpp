#include "guiData.h"
#include "utilities.h"
#include "main.h"
#include <algorithm>

GuiDataContainer::GuiDataContainer() :
    hide_gui(false),
    traced_depth(0),
    draw_coord_frame(false),
    draw_debug_aabb(false),
    draw_world_aabb(false),
    draw_GPU_tree(false),
    octree_depth(0),
    octree_depth_filter(-1),
    octree_intersection_cnt(0),
    test_tree(nullptr),
    desc(Denoiser::FilterType::ATROUS, glm::min(60, width), glm::ivec2(width, height), 0.5f, 0.5f, 0.5f)
{
    denoiser_options.is_on = false;
    denoiser_options.debug_tex_idx = 0;
}
void GuiDataContainer::ClearBuf() {
    char_bufs.clear();
}
char* GuiDataContainer::NextBuf() {
    char_bufs.emplace_back();
    return char_bufs.back().data;
}
void GuiDataContainer::Reset() {
    ClearBuf();
    denoiser_options.is_on = false;
    denoiser_options.debug_tex_idx = 0;

    draw_coord_frame = draw_world_aabb = draw_debug_aabb = false;
    octree_depth = 0;
    octree_depth_filter = -1;
    octree_intersection_cnt = 0;
    if (test_tree) {
        delete test_tree;
        test_tree = nullptr;
    }

    cur_scene.clear();
}
GuiDataContainer::~GuiDataContainer() {
    if (test_tree) {
        delete test_tree;
    }
}