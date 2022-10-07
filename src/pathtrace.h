#pragma once

#include <vector>
#include "scene.h"


class octreeGPU;
namespace PathTracer {
	void unitTest();
	void pathtraceInit(Scene* scene, RenderState* renderState, bool force_change = false);
	void pathtraceFree(Scene* scene, bool force_change = false);
	int pathtrace(uchar4* pbo, int iteration);
	bool saveRenderState(char const* filename);
	void togglePause();
	bool isPaused();
	octreeGPU getTree();
}