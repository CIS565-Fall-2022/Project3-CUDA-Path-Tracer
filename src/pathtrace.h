#pragma once

#include <vector>
#include "scene.h"

namespace PathTracer {
	void InitDataContainer(GuiDataContainer* guiData);
	void pathtraceInit(Scene* scene, RenderState* renderState);
	void pathtraceFree();
	int pathtrace(uchar4* pbo, int iteration);
	bool saveRenderState(char const* filename);
	void togglePause();
	bool isPaused();
}