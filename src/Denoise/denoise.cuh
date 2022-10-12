#pragma once
#include "denoise.h"

namespace Denoiser {
	// reference: https://jo.dreggn.org/home/2010_atrous.pdf
	__global__ void kern_denoise(
		color_t* out,
		color_t const* color_map,
		glm::vec3 const* norm_map,
		glm::vec3 const* pos_map,
		color_t const* diffuse_map,
		int step,
		ParamDesc const desc
	) {
		int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;
		int w = desc.res[0], h = desc.res[1];

		if (x >= w || y >= h)
			return;

#define FLATTEN(i, j) (i + (j * w))
		int idx = FLATTEN(x, y);
		color_t cval = color_map[idx];
		glm::vec3 nval = norm_map[idx];
		glm::vec3 pval = pos_map[idx];
		glm::vec3 sum(0, 0, 0);
		float cum_w = 0.f;
		for (size_t i = 0; i < 25; ++i) {
			int nx = glm::clamp(x + desc.offsets[i][0] * step, 0, w - 1);
			int ny = glm::clamp(y + desc.offsets[i][1] * step, 0, h - 1);
			
			idx = FLATTEN(nx, ny);
			color_t ctmp = color_map[idx];
			color_t t = cval - ctmp;

			float dist2 = glm::dot(t, t);
			float cw = glm::min(glm::exp(-dist2 / desc.c_phi), 1.f);
			glm::vec3 ntmp = norm_map[idx];
			t = nval - ntmp;

			dist2 = glm::max(glm::dot(t, t) / (step * step), 0.f);
			float nw = glm::min(glm::exp(-dist2 / desc.n_phi), 1.f);
			glm::vec3 ptmp = pos_map[idx];
			t = pval - ptmp;

			dist2 = glm::dot(t, t);
			float pw = glm::min(glm::exp(-dist2 / desc.p_phi), 1.f);
			float weight = cw * nw * pw;

			sum += ctmp * weight * desc.kernel[i];
			cum_w += weight * desc.kernel[i];
		}

		idx = FLATTEN(x, y);
		out[idx] = sum * diffuse_map[idx] / cum_w;
#undef FLATTEN
	}

	// wrapper that launches the denoiser
	void denoise(
		Span<color_t> rt,
		glm::vec3 const* norm_map,
		glm::vec3 const* pos_map,
		color_t const* diffuse_map,
		ParamDesc desc) {

		color_t* bufs[2];
		ALLOC(bufs[0], rt.size());
		ALLOC(bufs[1], rt.size());


		int buf_idx = 0;
		D2D(bufs[buf_idx], rt.get(), rt.size());

		const dim3 block_size(8, 8);
		const dim3 blocks_per_grid(
			DIV_UP(desc.res[0], block_size.x),
			DIV_UP(desc.res[1], block_size.y));

		bool flag = false;
		for (int step = 1; step <= desc.filter_size; step <<= 1, desc.c_phi /= 2) {
			kern_denoise KERN_PARAM(blocks_per_grid, block_size) (bufs[1 - buf_idx], bufs[buf_idx], norm_map, pos_map, diffuse_map, step, desc);
			checkCUDAError("denoise");
			buf_idx = 1 - buf_idx;
		}
		
		D2D(rt.get(), bufs[buf_idx], rt.size());

		FREE(bufs[0]);
		FREE(bufs[1]);
	}
}