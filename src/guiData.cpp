#include "guiData.h"
#include "utilities.h"
#include "main.h"
#include <algorithm>

GuiDataContainer::GuiDataContainer() :
    traced_depth(0),
    octree_depth(0),
    octree_depth_filter(-1),
    octree_intersection_cnt(0),
    test_tree(nullptr),
    desc(Denoiser::FilterType::ATROUS, glm::min(60, width), glm::ivec2(width, height), 0.5f, 0.5f, 0.5f),
    scene_file_dialog("Select Scene File", false, ".txt"),
    save_file_dialog("Select Save File", true, ".sav"),
    img_file_dialog("Select Image File", false),
    buf_id(0)
{
    denoiser_options.is_on = false;
    denoiser_options.debug_tex_idx = 0;
}
void GuiDataContainer::Reset() {
    for (int i = 0; i < bufs.size(); ++i) {
        memcpy(bufs[i].get(), default_bufs[i].second.get(), default_bufs[i].first);
    }
    ResetBuf();

    ref_img_file.clear();
    ref_img = nullptr;
    denoiser_options.is_on = false;
    denoiser_options.debug_tex_idx = 0;

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