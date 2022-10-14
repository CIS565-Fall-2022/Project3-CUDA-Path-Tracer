#pragma once
#include "denoise.h"
#include <thrust/transform.h>

namespace Denoiser {
	struct Demodulate {
		__device__ color_t operator()(color_t const& c, color_t const& albedo) const {
			return c / (albedo + EPSILON);
		}
	};
	struct Modulate {
		__device__ color_t operator()(color_t const& c, color_t const& albedo) const {
			return c * (albedo + EPSILON);
		}
	};

	struct Filter {
		struct Data {
			int nx, ny;
			float weight;
		};
		__device__ virtual Data operator*() const = 0;
		__device__ virtual void operator++() = 0;
		__device__ virtual operator bool() const = 0;
	};
	struct ATrousFilter : public Filter {
		int x, y, step;
		ParamDesc const& desc;

		// state
		// i : [-2, 2]
		// j : [-2, 2]
		int i, j;
		float weight;

		__device__ ATrousFilter(int x, int y, int step, ParamDesc const& desc) 
			: x(x), y(y), step(step), desc(desc), i(-2), j(-2), weight(1. / 16) { }
		__device__ virtual Data operator*() const {
			return {
				glm::clamp(x + i * step, 0, desc.res[0] - 1),
				glm::clamp(y + j * step, 0, desc.res[1] - 1),
				weight
			};
		}
		__device__ virtual void operator++() {
			if (j < 2) {
				++j;
			} else {
				++i;
				j = -2;
			}

			int x = glm::min(glm::abs(i), glm::abs(j));
			if (x == 0) {
				weight = 3. / 8;
			} else if (x == 1) {
				weight = 1. / 4;
			} else {
				weight = 1. / 16;
			}
		}
		__device__ virtual operator bool() const {
			return i <= 2;
		}
	};

	struct GaussianFilter : public Filter {
		int x, y;
		int si, ei;
		float two_d;
		ParamDesc const& desc;

		// state
		// i : [-w/2, w/2]
		// j : [-w/2, w/2]
		int i, j;
		float weight;

		__device__ GaussianFilter(int x, int y, int step, ParamDesc const& desc)
			: x(x), y(y), desc(desc), si(-step/2), ei(step/2), two_d(desc.s_dev * desc.s_dev) {
			i = si, j = si;
			weight = glm::exp(-(i * i + j * j) * 0.5f) / (2 * PI);
		}

		__device__ virtual Data operator*() const {
			return {
				glm::clamp(x + i, 0, desc.res[0] - 1),
				glm::clamp(y + j, 0, desc.res[1] - 1),
				weight
			};
		}

		__device__ virtual void operator++() {
			if (j < ei) {
				++j;
			} else {
				++i;
				j = si;
			}

			// sample weight from 2D Gaussian
			weight = glm::exp(-(i * i + j * j) / (2 * two_d)) / (2 * PI * two_d);
		}

		__device__ virtual operator bool() const {
			return i <= ei;
		}
	};


	// reference: https://jo.dreggn.org/home/2010_atrous.pdf
	template<typename FilterT>
	__global__ void kern_denoise(
		color_t* out,
		color_t const* color_map,
		glm::vec3 const* norm_map,
		glm::vec3 const* pos_map,
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
		for (FilterT filter{ x,y,step,desc }; filter; ++filter) {
			auto data = *filter;
			
			idx = FLATTEN(data.nx, data.ny);
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

			sum += ctmp * weight * data.weight;
			cum_w += weight * data.weight;
		}

		idx = FLATTEN(x, y);
		out[idx] = sum / cum_w;
#undef FLATTEN
	}

	// wrapper that launches the denoiser
	void denoise(
		Span<color_t> rt,
		glm::vec3 const* norm_map,
		glm::vec3 const* pos_map,
		color_t const* diffuse_map,
		ParamDesc desc
	) {

		int buf_idx = 0;
		color_t* bufs[2];

		bufs[buf_idx] = rt;
		ALLOC(bufs[1 - buf_idx], rt.size());

#ifdef DENOISE_USE_DIFFUSE_MAP
		// divide by diffuse map, i.e. demodulate
		thrust::transform(
			thrust::device,
			bufs[buf_idx],
			bufs[buf_idx] + rt.size(),
			diffuse_map,
			bufs[buf_idx],
			Demodulate());
#endif

		const dim3 block_size(8, 8);
		const dim3 blocks_per_grid(
			DIV_UP(desc.res[0], block_size.x),
			DIV_UP(desc.res[1], block_size.y));

		switch (desc.type) {
		case FilterType::ATROUS:
			for (int step = 1; step <= desc.filter_size; step <<= 1, desc.c_phi /= 2) {
				kern_denoise<ATrousFilter> KERN_PARAM(blocks_per_grid, block_size) (bufs[1 - buf_idx], bufs[buf_idx], norm_map, pos_map, step, desc);
				checkCUDAError("denoise");
				buf_idx = 1 - buf_idx;
			}
			break;
		case FilterType::GAUSSIAN:
			for (int step = 3; step <= desc.filter_size; step <<= 1, desc.c_phi /= 2) {
				kern_denoise<GaussianFilter> KERN_PARAM(blocks_per_grid, block_size) (bufs[1 - buf_idx], bufs[buf_idx], norm_map, pos_map, step, desc);
				checkCUDAError("denoise");
				buf_idx = 1 - buf_idx;
			}
			break;
		default:
			break;
		}

		
#ifdef DENOISE_USE_DIFFUSE_MAP
		// multiply by diffuse map, i.e. modulate
		thrust::transform(
			thrust::device,
			bufs[buf_idx],
			bufs[buf_idx] + rt.size(),
			diffuse_map,
			bufs[buf_idx],
			Modulate());
#endif
		if (bufs[buf_idx] != rt) {
			D2D(rt.get(), bufs[buf_idx], rt.size());
		}

		// avoid freeing supplied pointer
		if (bufs[0] != rt) {
			FREE(bufs[0]);
		}
		if (bufs[1] != rt) {
			FREE(bufs[1]);
		}
	}
}