#pragma once
#include <cuda.h>
#include <glm/glm.hpp>
#include "../utilities.h"
#include "../sceneStructs.h"

namespace Denoiser {
	struct ParamDesc {
		int filter_size;
		int step;
		glm::ivec2 offsets[25];
		float kernel[25];
		glm::ivec2 res;
		float c_phi, n_phi, p_phi;
	};

	// reference: https://jo.dreggn.org/home/2010_atrous.pdf
	__global__ void denoise(
		color_t* out,
		color_t const* color_map,
		glm::vec3 const* norm_map,
		glm::vec3 const* pos_map,
		ParamDesc const desc
	) {
		int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;
		int w = desc.res[0], h = desc.res[1];

#define FLATTEN(i, j, ncol) (i * ncol + j)
		int idx = FLATTEN(x, y, h);
		color_t cval = color_map[idx];
		glm::vec3 nval = norm_map[idx];
		glm::vec3 pval = pos_map[idx];
		glm::vec3 sum{ 0 };
		float cum_w = 0.f;
		for (size_t i = 0; i < 25; ++i) {
			int nx = glm::clamp(x + desc.offsets[i][0] * desc.step, 0, desc.res[0]);
			int ny = glm::clamp(y + desc.offsets[i][1] * desc.step, 0, desc.res[1]);
			
			idx = FLATTEN(nx, ny, h);
			color_t ctmp = color_map[idx];
			color_t t = cval - ctmp;

			float dist2 = glm::dot(t, t);
			float cw = glm::min(glm::exp(-(dist2) / desc.c_phi), 1.f);
			glm::vec3 ntmp = norm_map[idx];
			t = nval - ntmp;

			dist2 = glm::max(glm::dot(t, t) / desc.step * desc.step, 0.f);
			float nw = glm::min(glm::exp(-dist2 / desc.n_phi), 1.f);
			glm::vec3 ptmp = pos_map[idx];
			t = pval - ptmp;

			dist2 = glm::dot(t, t);
			float pw = glm::min(glm::exp(-dist2 / desc.p_phi), 1.f);
			float weight = cw * nw * pw;

			sum += ctmp * weight * desc.kernel[i];
			cum_w += weight * desc.kernel[i];
		}

		idx = FLATTEN(x, y, h);
		out[idx] = sum / cum_w;
#undef FLATTEN
	}
}