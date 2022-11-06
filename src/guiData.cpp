#include "guiData.h"
#include "utilities.h"
#include "main.h"
#include "ImGui/imgui.h"
#include <algorithm>

GuiDataContainer::GuiDataContainer() :
    traced_depth(0),
    octree_depth(0),
    octree_depth_filter(-1),
    octree_intersection_cnt(0),
    test_tree(nullptr),
    desc(Denoiser::FilterType::ATROUS, glm::min(60, width), glm::ivec2(width, height), 0.5f, 0.5f, 0.5f)
{
    denoiser_options.is_on = false;
    denoiser_options.debug_tex_idx = 0;
}
void GuiDataContainer::Reset() {
    for (auto& kvp : bufs) {
        memcpy(kvp.second.data.get(), kvp.second.default_data.get(), kvp.second.size_bytes);
    }
    desc = Denoiser::ParamDesc(Denoiser::FilterType::ATROUS, glm::min(60, width), glm::ivec2(width, height), 0.5f, 0.5f, 0.5f);
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

std::string GuiDataContainer::OpenFileDialogue(char const* label, bool dirmode, char const* ext) {
    std::string key = label;
    std::string ret = ImGui::OpenFileDialogue(*GetOrSetBuf<ImGui::FileDialogue>(key, label, dirmode, ext), label);
    if (ret.size()) {
        GetOrSetBuf<std::string>(key + "_string", ret);
    }
    return ret;
}

bool GuiDataContainer::CheckBox(char const* label) {
    auto ptr = GetOrSetBuf<bool>(label);
    ImGui::Checkbox(label, ptr.get());
    return *ptr;
}

bool GuiDataContainer::ToggleButton(char const* label_if_true, char const* label_if_false) {
    auto ptr = GetOrSetBuf<bool>(label_if_true);
    if (!ptr) {
        if (ImGui::Button(label_if_false)) {
            *ptr = true;
        }
    } else {
        if (ImGui::Button(label_if_true)) {
            *ptr = false;
        }
    }
    return *ptr;
}