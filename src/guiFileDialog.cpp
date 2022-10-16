#include "guiFileDialog.h"
#include "ImGui/imgui.h"
#include <string>
#include <vector>
#include <iostream>

namespace fs = std::experimental::filesystem;
using FileDialogue = ImGui::FileDialogue;

struct Scope {
	void (*call)();
	Scope(void(*call)()) : call(call) { }
	~Scope() { call(); }
	Scope(Scope&&) = delete;
	Scope(Scope const&) = delete;
};

// are we selecting a directory?
bool FileDialogue::IsDirectoryMode() const {
	return _dirmode;
}

void FileDialogue::SetPath(fs::path const& newPath) {
	_result.clear();
	_cur_path = newPath;
}

void FileDialogue::DoWindow() {
	if (!_is_open) {
		return;
	}
	Scope _{ImGui::End};
	if (ImGui::Begin(_name, &_is_open, ImGuiWindowFlags_AlwaysAutoResize)) {

		if (IsDirectoryMode()) {
			ImGui::InputText("Enter File Name", _buf, 256);
		}

		ImGui::BeginDisabled(!FileDialogue::HasResult());
		if (ImGui::Button("Confirm")) {
			_is_open = false;
			_finished = true;
			ImGui::EndDisabled();
			return;
		}
		ImGui::EndDisabled();

		if (_cur_path.has_parent_path() && ImGui::Selectable("..")) { // go to parent
			SetPath(_cur_path.parent_path());
			return;
		}

		for (auto const& entry : fs::directory_iterator(_cur_path)) {
			auto const& path = entry.path();
			if (!_ext.empty() && !fs::is_directory(path) && path.extension() != _ext) {
				continue;
			}

			std::string name = path.filename().string();

			if (fs::is_directory(path)) {
				name = '[' + name + ']';

				ImGui::Selectable(name.c_str());
				if (IsDirectoryMode()) {
					// double click to enter, click to select
					if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
						SetPath(path);
						return;
					} else if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
						_result = path;
					}
				} else {
					// click to enter
					if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
						SetPath(path);
						return;
					}
				}
			} else {
				if (ImGui::Selectable(name.c_str())) {
					_result = path;
				}
			}
		}
	}
}

void FileDialogue::Reset() {
	_finished = false;
	_cur_path = fs::current_path();
	_result.clear();
}

bool FileDialogue::IsOpen() const {
	return _is_open;
}

bool FileDialogue::HasResult() const {
	if (IsDirectoryMode()) {
		return fs::is_directory(_result) && strlen(_buf);
	}
	return _result.has_extension() && _result.has_filename() && _result.extension() == _ext;
}
bool FileDialogue::Finished() const {
	return HasResult() && _finished;
}
std::string FileDialogue::GetPath() const {
	auto ret = _result;
	if (IsDirectoryMode()) {
		ret /= _buf;
		return ret.string() + _ext;
	}
	return ret.string();
}

FileDialogue::FileDialogue(char const* name, bool dirmode, char const* ext)
	: _name(name), _dirmode(dirmode), _is_open(false), _finished(false) {
	memset(_buf, 0, sizeof(_buf));
	if (ext && strlen(ext)) _ext = ext;
}

void FileDialogue::FileDialogButton(char const* label) {
	if (IsOpen()) {
		if (ImGui::Button(label)) {
			ImGui::SetWindowFocus(_name);
		}
	} else {
		if (ImGui::Button(label)) {
			Reset();
			_is_open = true;
		}
	}
	DoWindow();
}

bool ImGui::OpenFileDialogue(FileDialogue& dialogue, char const* label, std::string& result) {
	ImGui::Text(label);
	ImGui::SameLine();
	std::string id = "select file";
	id += "##";
	id += dialogue._name;
	dialogue.FileDialogButton(id.c_str());
	if (dialogue.Finished()) {
		result = dialogue.GetPath();
		dialogue.Reset();
		return true;
	}
	return false;
}