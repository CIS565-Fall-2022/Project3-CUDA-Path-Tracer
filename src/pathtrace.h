#pragma once

#include <vector>
#include "scene.h"
#include "Profile/timer.h"

struct octreeGPU;
namespace Denoiser {
	struct ParamDesc;
}

enum DebugTextureType {
	NONE,
	NORM_BUF,
	POS_BUF,
	DIFFUSE_BUF,
	NUM_OPTIONS
};

namespace PathTracer {
	void unitTest();
	void pathtraceInit(Scene* scene, RenderState* state, bool force_change = false);
	void pathtraceFree(Scene* scene, bool force_change = false);
	int pathtrace(int iteration);
	bool saveRenderState(char const* filename);
	void togglePause();
	void enableDenoise();
	void disableDenoise();

	bool isPaused();
	octreeGPU getTree();

	void setDenoise(Denoiser::ParamDesc const& param);
	void debugTexture(DebugTextureType type);

	void beginFrame(unsigned int pbo_id);
	void endFrame();

	std::unordered_map<std::string, Profiling::ProfileData>& GetProfileData();
}