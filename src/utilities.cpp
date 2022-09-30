//  UTILITYCORE- A Utility Library by Yining Karl Li
//  This file is part of UTILITYCORE, Copyright (c) 2012 Yining Karl Li
//
//  File: utilities.cpp
//  A collection/kitchen sink of generally useful functions

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <iostream>
#include <cstdio>
#include "main.h"
#include "utilities.h"

#ifdef _WIN32
#include <Windows.h>
#endif // _WIN32

static std::vector<std::string> getFilesInDir(char const* dir) {
    // credit: https://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c

    std::vector<std::string> names;
#ifdef _WIN32
    std::string search_pattern(dir);
    std::string search_path(dir);
    search_pattern += "*.*";

    WIN32_FIND_DATA fd;
    HANDLE hFind = ::FindFirstFile(search_pattern.c_str(), &fd);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            // read all (real) files in current folder
            // , delete '!' read other 2 default folder . and ..
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                names.emplace_back(search_path + fd.cFileName);
            }
        } while (::FindNextFile(hFind, &fd));
        ::FindClose(hFind);
    }
#else
    assert(!"unsupported platform")
#endif
    return names;
}

GuiDataContainer::GuiDataContainer() :
    traced_depth(0), cur_scene(0), cur_save(0), prompt_text("") {
    memset(buf, 0, sizeof(buf));

    auto scene_files = getFilesInDir(scene_files_dir);
    auto save_files = getFilesInDir(save_files_dir);

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
GuiDataContainer::~GuiDataContainer() {
    for (int i = 0; i < num_scenes; ++i) {
        delete scene_file_names[i];
    }
    for (int i = 0; i < num_saves; ++i) {
        delete save_file_names[i];
    }
    delete[] scene_file_names;
    delete[] save_file_names;
}
float utilityCore::clamp(float f, float min, float max) {
    if (f < min) {
        return min;
    } else if (f > max) {
        return max;
    } else {
        return f;
    }
}

bool utilityCore::replaceString(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if (start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

std::string utilityCore::convertIntToString(int number) {
    std::stringstream ss;
    ss << number;
    return ss.str();
}

glm::vec3 utilityCore::clampRGB(glm::vec3 color) {
    if (color[0] < 0) {
        color[0] = 0;
    } else if (color[0] > 255) {
        color[0] = 255;
    }
    if (color[1] < 0) {
        color[1] = 0;
    } else if (color[1] > 255) {
        color[1] = 255;
    }
    if (color[2] < 0) {
        color[2] = 0;
    } else if (color[2] > 255) {
        color[2] = 255;
    }
    return color;
}

bool utilityCore::epsilonCheck(float a, float b) {
    if (fabs(fabs(a) - fabs(b)) < EPSILON) {
        return true;
    } else {
        return false;
    }
}

glm::mat4 utilityCore::buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) {
    glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
    glm::mat4 rotationMat =   glm::rotate(glm::mat4(), rotation.x * (float) PI / 180, glm::vec3(1, 0, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * (float) PI / 180, glm::vec3(0, 1, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * (float) PI / 180, glm::vec3(0, 0, 1));
    glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
    return translationMat * rotationMat * scaleMat;
}

std::vector<std::string> utilityCore::tokenizeString(std::string str) {
    std::stringstream strstr(str);
    std::istream_iterator<std::string> it(strstr);
    std::istream_iterator<std::string> end;
    std::vector<std::string> results(it, end);
    return results;
}

std::istream& utilityCore::safeGetline(std::istream& is, std::string& t) {
    t.clear();

    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks,
    // such as thread synchronization and updating the stream state.

    std::istream::sentry se(is, true);
    std::streambuf* sb = is.rdbuf();

    for (;;) {
        int c = sb->sbumpc();
        switch (c) {
        case '\n':
            return is;
        case '\r':
            if (sb->sgetc() == '\n')
                sb->sbumpc();
            return is;
        case EOF:
            // Also handle the case when the last line has no line ending
            if (t.empty())
                is.setstate(std::ios::eofbit);
            return is;
        default:
            t += (char)c;
        }
    }
}


std::istream& utilityCore::peekline(std::istream& is, std::string& t) {
    t.clear();
    int pos = is.tellg();
    safeGetline(is, t);
    is.seekg(pos, std::ios_base::beg);
    return is;
}