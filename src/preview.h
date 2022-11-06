#pragma once

extern GLuint pbo;
struct GuiDataContainer;


namespace Preview {
	GuiDataContainer* GetGUIData();
	bool initImguiGL();
	bool initBufs();
	void mainLoop();
	bool CapturingMouse();
	bool CapturingKeyboard();
	void InitImguiData();
	void DoPreloadMenu();
}