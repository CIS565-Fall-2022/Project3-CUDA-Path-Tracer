#pragma once
#include "Denoise/denoise.h"
#include "guiFileDialog.h"
#include <vector>
#include <unordered_map>
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

    std::string img_data_file;
    std::string ref_img_file;
    std::unique_ptr<Image> ref_img;

    std::string cur_scene;

    struct {
        bool is_on;
        int debug_tex_idx;
    } denoiser_options;

    struct BufInfo {
        size_t size_bytes;
        std::shared_ptr<void> data;
        std::shared_ptr<void> default_data;
    };

    // this allows us to create bufs in a completely generic way
    std::unordered_map<std::string, BufInfo> bufs;

    /// <summary>
    /// registers a buffer with a string id "key"
    /// </summary>
    template<typename T, typename... Args>
    std::shared_ptr<T> GetOrSetBuf(std::string const& key, Args&&... args) {
        if (!bufs.count(key)) {
            BufInfo info;
            info.size_bytes = sizeof(T);
            info.data = std::make_shared<T>(std::forward<Args>(args)...);
            info.default_data = std::make_shared<T>(*std::static_pointer_cast<T>(info.data));
            bufs[key] = std::move(info);
        }
        return std::static_pointer_cast<T>(bufs[key].data);
    }

    // convenience wrappers
    std::string OpenFileDialogue(char const* label, bool dirmode, char const* ext);
    bool CheckBox(char const* label);
    bool ToggleButton(char const* label_if_true, char const* label_if_false);
};