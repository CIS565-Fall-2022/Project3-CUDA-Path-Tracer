#pragma once
#include "denoise.h"

#include <glm/gtx/transform.hpp>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

namespace Denoiser {
	class DenoiseBuffers {
		int pixelcount;
		glm::vec3* n;
		glm::vec3* x;
		glm::vec3* d; // no need to make these spans because sizes are all the same

	public:
		DenoiseBuffers() = default;
		DenoiseBuffers(DenoiseBuffers const& o) = default;
		DenoiseBuffers(DenoiseBuffers&&) = delete;

		void init(int pixelcount) {
			this->pixelcount = pixelcount;
			d = make_span<glm::vec3>(pixelcount);
			n = make_span<glm::vec3>(pixelcount);
			x = make_span<glm::vec3>(pixelcount);
		}
		void free() {
			FREE(n);
			FREE(x);
			FREE(d);
		}
		void set(ShadeableIntersection const* dev_inters, Material const* materials) {
#ifdef DENOISE_GBUF_OPTIMIZATION
#else
			// normal
			thrust::transform(
				thrust::device,
				dev_inters,
				dev_inters + pixelcount,
				n,
				Denoiser::IntersectionToNormal());

			// position
			thrust::transform(
				thrust::device,
				dev_inters,
				dev_inters + pixelcount,
				x,
				Denoiser::IntersectionToPos());

			// diffuse
			thrust::transform(
				thrust::device,
				dev_inters,
				dev_inters + pixelcount,
				d,
				Denoiser::IntersectionToDiffuse(materials));
#endif
		}
		__host__ __device__ int size() const {
			return pixelcount;
		}
		__host__ __device__ glm::vec3 const* get_normal() const {
			return n;
		}
		__host__ __device__ glm::vec3 const* get_pos() const {
			return x;
		}
		__host__ __device__ glm::vec3 const* get_diffuse() const {
			return d;
		}
		__host__ __device__ glm::vec3 get_normal(int i) const {
#ifdef DENOISE_GBUF_OPTIMIZATION
#else
			return n[i];
#endif
		}
		__host__ __device__ glm::vec3 get_pos(int i) const {
#ifdef DENOISE_GBUF_OPTIMIZATION
			
#else
			return x[i];
#endif
		}
		__host__ __device__ glm::vec3 get_diffuse(int i) const {
#ifdef DENOISE_GBUF_OPTIMIZATION
#else
			return d[i];
#endif
		}
	};

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
	__global__ void kern_denoise(color_t* out, color_t* in, DenoiseBuffers gbuf, int step, ParamDesc desc) {
		int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;
		int w = desc.res[0], h = desc.res[1];

		if (x >= w || y >= h)
			return;

#define FLATTEN(i, j) (i + (j * w))
		int idx = FLATTEN(x, y);
		color_t cval = in[idx];
		glm::vec3 nval = gbuf.get_normal(idx);
		glm::vec3 pval = gbuf.get_pos(idx);
		glm::vec3 sum(0, 0, 0);
		float cum_w = 0.f;
		for (FilterT filter{ x,y,step,desc }; filter; ++filter) {
			auto data = *filter;
			
			idx = FLATTEN(data.nx, data.ny);
			color_t ctmp = in[idx];
			color_t t = cval - ctmp;

			float dist2 = glm::dot(t, t);
			float cw = glm::min(glm::exp(-dist2 / desc.c_phi), 1.f);
			glm::vec3 ntmp = gbuf.get_normal(idx);
			t = nval - ntmp;

			dist2 = glm::max(glm::dot(t, t) / (step * step), 0.f);
			float nw = glm::min(glm::exp(-dist2 / desc.n_phi), 1.f);
			glm::vec3 ptmp = gbuf.get_pos(idx);
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
	void denoise(color_t* denoise_image, DenoiseBuffers gbuf, ParamDesc desc) {
		int buf_idx = 0;
		color_t* bufs[2];

		bufs[buf_idx] = denoise_image;
		ALLOC(bufs[1 - buf_idx], gbuf.size());

#ifdef DENOISE_USE_DIFFUSE_MAP
		// divide by diffuse map, i.e. demodulate
		thrust::transform(
			thrust::device,
			bufs[buf_idx],
			bufs[buf_idx] + gbuf.size(),
			gbuf.get_diffuse(),
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
				kern_denoise<ATrousFilter> KERN_PARAM(blocks_per_grid, block_size) (bufs[1 - buf_idx], bufs[buf_idx], gbuf, step, desc);
				checkCUDAError("denoise");
				buf_idx = 1 - buf_idx;
			}
			break;
		case FilterType::GAUSSIAN:
			for (int step = 3; step <= desc.filter_size; step <<= 1, desc.c_phi /= 2) {
				kern_denoise<GaussianFilter> KERN_PARAM(blocks_per_grid, block_size) (bufs[1 - buf_idx], bufs[buf_idx], gbuf, step, desc);
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
			bufs[buf_idx] + gbuf.size(),
			gbuf.get_diffuse(),
			bufs[buf_idx],
			Modulate());
#endif
		if (bufs[buf_idx] != denoise_image) {
			D2D(denoise_image, bufs[buf_idx], gbuf.size());
		}

		// avoid freeing supplied pointer
		if (bufs[0] != denoise_image) {
			FREE(bufs[0]);
		}
		if (bufs[1] != denoise_image) {
			FREE(bufs[1]);
		}
	}
}