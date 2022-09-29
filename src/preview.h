#pragma once

extern GLuint pbo;

std::string currentTimeString();
bool initImguiGL();
bool init();
void mainLoop();
void resetImguiState();
bool MouseOverImGuiWindow();
void InitImguiData(GuiDataContainer* guiData);
void DoPreloadMenu();