#include "guiData.h"
#include "utilities.h"
#include "main.h"
#include <algorithm>

GuiDataContainer::GuiDataContainer() :
    hide_gui(false),
    traced_depth(0),
    cur_scene(0),
    cur_save(0),
    prompt_text(""),
    draw_coord_frame(false),
    draw_debug_aabb(false),
    draw_world_aabb(false),
    draw_GPU_tree(false),
    octree_depth(0),
    octree_depth_filter(-1),
    octree_intersection_cnt(0),
    test_tree(nullptr),
    desc(glm::min(60, width), glm::ivec2(width, height), 0.5f, 0.5f, 0.5f)
{
    denoiser_options.is_on = false;
    denoiser_options.debug_tex_idx = 0;

    memset(save_file_name_buf, 0, sizeof(save_file_name_buf));

    auto scene_files = utilityCore::getFilesInDir(scene_files_dir.c_str());
    auto save_files = utilityCore::getFilesInDir(save_files_dir.c_str());

    num_scenes = scene_files.size();
    num_saves = save_files.size();

    scene_file_names = new char* [num_scenes];
    save_file_names = new char* [num_saves];

    for (int i = 0; i < num_scenes; ++i) {
        scene_file_names[i] = new char[scene_files[i].size() + 1];
        strcpy(scene_file_names[i], scene_files[i].c_str());
    }

    for (int i = 0; i < num_saves; ++i) {
        save_file_names[i] = new char[save_files[i].size() + 1];
        strcpy(save_file_names[i], save_files[i].c_str());
    }
}
void GuiDataContainer::Reset() {
    denoiser_options.is_on = false;
    denoiser_options.debug_tex_idx = 0;

    memset(save_file_name_buf, 0, sizeof(save_file_name_buf));

    prompt_text = "";
    draw_coord_frame = draw_world_aabb = draw_debug_aabb = false;
    octree_depth = 0;
    octree_depth_filter = -1;
    octree_intersection_cnt = 0;
    if (test_tree) {
        delete test_tree;
        test_tree = nullptr;
    }
}
GuiDataContainer::~GuiDataContainer() {
    for (int i = 0; i < num_scenes; ++i) {
        delete scene_file_names[i];
    }
    for (int i = 0; i < num_saves; ++i) {
        delete save_file_names[i];
    }
    delete[] scene_file_names;
    delete[] save_file_names;
    if (test_tree) {
        delete test_tree;
    }
}