#pragma once
#include "Denoise/denoise.h"
#include "guiFileDialog.h"
#include <vector>
#include <string>

class octree;
class Image;

class GuiDataContainer
{
public:
    static constexpr int BUF_SIZE = 256;
    GuiDataContainer();
    ~GuiDataContainer();
    void Reset();

    int traced_depth;
    int octree_depth;
    int octree_depth_filter;
    int octree_intersection_cnt;
    octree* test_tree;
    Denoiser::ParamDesc desc;
    
    ImGui::FileDialogue scene_file_dialog;
    ImGui::FileDialogue save_file_dialog;
    ImGui::FileDialogue img_file_dialog;
    ImGui::FileDialogue img_file_data_dialog;

    std::string img_data_file;
    std::string ref_img_file;
    std::unique_ptr<Image> ref_img;

    std::string cur_scene;

    struct {
        bool is_on;
        int debug_tex_idx;
    } denoiser_options;

    // this allows us to create bufs in a completely generic way
    std::vector<std::shared_ptr<void>> bufs;
    std::vector<std::pair<int, std::shared_ptr<void>>> default_bufs;
    int buf_id;
    void ResetBuf() { buf_id = 0; }
    template<typename T, typename... Args>
    void NextBuf(Args&&... args) {
        if (buf_id >= bufs.size()) {
            bufs.emplace_back(new T(std::forward<Args>(args)...));
            default_bufs.emplace_back(sizeof(T), new T(*std::static_pointer_cast<T>(bufs.back())));
        }
        ++buf_id;
    }

    template<typename T>
    std::shared_ptr<T> CurBuf() const { 
        return std::static_pointer_cast<T>(bufs[buf_id-1]);
    }
    template<typename T>
    T& CurBufData() const {
        return *CurBuf<T>();
    }

    void OpenFileDialogue(char const* label, bool dirmode, char const* ext);
};