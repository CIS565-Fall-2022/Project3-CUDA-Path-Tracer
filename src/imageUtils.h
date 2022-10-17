#pragma once
#include <string>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include <vector_types.h>

#include "image.h"

struct RenderState;

namespace ImageUtils {
	void SaveImage(glm::vec3 const* src, std::string const& name, bool radiance);
	void SaveImage(RenderState const* state);
	void SaveImage(uchar4 const* pbo);

	float CalculateMSE(int size, glm::vec3 const* img1, glm::vec3 const* img2);
	float CalculatePSNR(int size, glm::vec3 const* img1, glm::vec3 const* img2);
}