#pragma once
#include "denoise.h"

#ifdef DENOISE_GBUF_OPTIMIZATION
#include "../sceneStructs.h"
#include "../camState.h"
#endif // DENOISE_GBUF_OPTIMIZATION

#include <glm/gtx/transform.hpp>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

namespace Denoiser {
#ifdef DENOISE_GBUF_OPTIMIZATION
	struct NormPos {
		float2 encoded_norm;
		float  pos_depth; // depth in clip space
	};
	struct EncodeNormPos {
		glm::mat4x4 view, proj;
		__host__ __device__ EncodeNormPos(glm::mat4x4 const& view, glm::mat4x4 const& proj) : view(view), proj(proj) { }
		__device__ NormPos operator()(ShadeableIntersection const& inters) const {
			if (inters.t < 0) {
				return { 0, 0, 0 };
			}
			glm::vec4 tmp = proj * (view * glm::vec4(inters.hitPoint, 1.f));
			tmp /= tmp.w;
			return { inters.surfaceNormal.x, inters.surfaceNormal.y, tmp.z };
		}
	};
	struct DecodeNorm {
		__device__ glm::vec3 operator()(NormPos const& xn) const {
			float x = xn.encoded_norm.x;
			float y = xn.encoded_norm.y;
			float z = glm::sqrt(1.f - x * x - y * y);
			return glm::vec3(x, y, z);
		}
	};
	struct DecodeNormRGBA {
		__device__ uchar4 operator()(NormPos const& xn) const {
			return NormalToRGBA()(DecodeNorm()(xn));
		}
	};
	struct DecodePos {
		glm::mat4x4 inv_view, inv_proj;
		float x, y;
		__host__ __device__ DecodePos(
			glm::mat4x4 const& inv_view, 
			glm::mat4x4 const& inv_proj, 
			glm::ivec2 const& viewport_coord, 
			glm::ivec2 const& res)
			: inv_view(inv_view), inv_proj(inv_proj) {
			x = viewport_coord.x / (float)res.x;
			y = viewport_coord.y / (float)res.y;

			x = x * 2 - 1;
			y = (1 - y) * 2 - 1;
		}
		__device__ glm::vec3 operator()(NormPos const& xn) {
			glm::vec4 pos4 = inv_proj * glm::vec4(x, y, xn.pos_depth, 1);
			pos4 /= pos4.w;

			return glm::vec3(inv_view * pos4);
		}
	};
	struct DecodePosRGBA {
		glm::mat4x4 inv_view, inv_proj;
		glm::ivec2 viewport_coord;
		glm::ivec2 res;
		__host__ __device__ DecodePosRGBA(
			glm::mat4x4 const& inv_view,
			glm::mat4x4 const& inv_proj,
			glm::ivec2 const& viewport_coord,
			glm::ivec2 const& res)
			: inv_view(inv_view), inv_proj(inv_proj), viewport_coord(viewport_coord), res(res) { }
		__device__ uchar4 operator()(NormPos const& xn) {
			return NormalizedRGBToRGBA()(DecodePos(inv_view, inv_proj, viewport_coord, res)(xn));
		}
	};
#endif

	class DenoiseBuffers {
		int pixelcount;
		int w, h;

		// no need to make these spans because sizes are all the same
#ifdef DENOISE_GBUF_OPTIMIZATION
		using vec_type = NormPos;
		
		NormPos* xn;
		glm::mat4x4 inv_view, inv_proj;
#else
		using vec_type = glm::vec3;

		glm::vec3* n, *x;
#endif
		glm::vec3* d;

	public:
		DenoiseBuffers() = default;
		DenoiseBuffers(DenoiseBuffers const& o) = default;
		DenoiseBuffers(DenoiseBuffers&&) = delete;

		void init(int w, int h) {
			this->w = w;
			this->h = h;
			this->pixelcount = w * h;
			d = make_span<glm::vec3>(pixelcount);

#ifdef DENOISE_GBUF_OPTIMIZATION
			inv_view = glm::inverse(CamState::get_view());
			inv_proj = glm::inverse(CamState::get_proj());

			xn = make_span<NormPos>(pixelcount);
#else
			n = make_span<glm::vec3>(pixelcount);
			x = make_span<glm::vec3>(pixelcount);
#endif
		}
		void free() {
			FREE(d);
#ifdef DENOISE_GBUF_OPTIMIZATION
			FREE(xn);
#else
			FREE(n);
			FREE(x);
#endif
		}
		void set(ShadeableIntersection const* dev_inters, Material const* materials) {
#ifdef DENOISE_GBUF_OPTIMIZATION
			thrust::transform(
				thrust::device,
				dev_inters,
				dev_inters + pixelcount,
				xn,
				Denoiser::EncodeNormPos(CamState::get_view(), CamState::get_proj())
			);

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
#endif

			// diffuse
			thrust::transform(
				thrust::device,
				dev_inters,
				dev_inters + pixelcount,
				d,
				Denoiser::IntersectionToDiffuse(materials));
		}
		__host__ __device__ int size() const {
			return pixelcount;
		}
		__host__ __device__ vec_type const* get_normal() const {
#ifdef DENOISE_GBUF_OPTIMIZATION
			return xn;
#else
			return n;
#endif
		}
		__host__ __device__ vec_type const* get_pos() const {
#ifdef DENOISE_GBUF_OPTIMIZATION
			return xn;
#else
			return x;
#endif
		}
		__host__ __device__ glm::vec3 const* get_diffuse() const {
			return d;
		}
		__device__ glm::vec3 get_normal(int i) const {
#ifdef DENOISE_GBUF_OPTIMIZATION
			return DecodeNorm()(xn[i]);
#else
			return n[i];
#endif
		}
		__device__ glm::vec3 get_pos(int i) const {
#ifdef DENOISE_GBUF_OPTIMIZATION
			int x = i % w;
			int y = i / w;
			return DecodePos(inv_view, inv_proj, glm::ivec2(x, y), glm::ivec2(w, h))(xn[i]);
#else
			return x[i];
#endif
		}
		__device__ glm::vec3 get_diffuse(int i) const {
			return d[i];
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


	// processes a tile of TILE_SIZE x TILE_SIZE elements
	// launch parameters:
	// TILE_SIZE x TILE_SIZE threads per block
	// DIV_UP(w, TILE_SIZE) x DIV_UP(h, TILE_SIZE) blocks

	template<size_t TILE_SIZE_X, size_t TILE_SIZE_Y>
	__global__ void kern_atrous_denoise_shared(color_t* out, color_t* in, DenoiseBuffers gbuf, int step, ParamDesc desc) {
		__shared__ struct { 
			color_t color;
			glm::vec3 normal, pos;
			float weight;
		} smem_in[TILE_SIZE_X][TILE_SIZE_Y][25];

		static constexpr float kernels[3] = { 3. / 8, 1. / 4, 1. / 16 };

		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int w = desc.res[0], h = desc.res[1];

		// set up shared memory
		if (x < w && y < h) {
			for (int j = -2, k = 0; j <= 2; ++j) {
				for (int i = -2; i <= 2; ++i) {
					int idx = glm::clamp(x + i * step, 0, w - 1) + w * glm::clamp(y + j * step, 0, h - 1);

					auto& data = smem_in[threadIdx.x][threadIdx.y][k];
					data.color = in[idx];
					data.normal = gbuf.get_normal(idx);
					data.pos = gbuf.get_pos(idx);
					data.weight = kernels[glm::min(glm::abs(i), glm::abs(j))];
					
					++k;
				}
			}
		} else {
			return;
		}

		int idx = x + y * w;
		color_t cval = in[idx];
		glm::vec3 nval = gbuf.get_normal(idx);
		glm::vec3 pval = gbuf.get_pos(idx);
		glm::vec3 sum(0, 0, 0);
		float cum_w = 0.f;

#pragma unroll 25
		for (int i = 0; i < 25; ++i) {
			auto const& data = smem_in[threadIdx.x][threadIdx.y][i];
			color_t ctmp = data.color;
			color_t t = cval - ctmp;

			float dist2 = glm::dot(t, t);
			float cw = glm::min(glm::exp(-dist2 / desc.c_phi), 1.f);
			glm::vec3 ntmp = data.normal;
			t = nval - ntmp;

			dist2 = glm::max(glm::dot(t, t) / (step * step), 0.f);
			float nw = glm::min(glm::exp(-dist2 / desc.n_phi), 1.f);
			glm::vec3 ptmp = data.pos;
			t = pval - ptmp;

			dist2 = glm::dot(t, t);
			float pw = glm::min(glm::exp(-dist2 / desc.p_phi), 1.f);
			float weight = cw * nw * pw;

			sum += ctmp * weight * data.weight;
			cum_w += weight * data.weight;
		}
		out[idx] = sum / cum_w;
	}

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

	__global__ void kern_denoise_pure_blur(color_t* out, color_t* in, int step, ParamDesc desc) {
		int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;
		int w = desc.res[0], h = desc.res[1];

		if (x >= w || y >= h)
			return;

#define FLATTEN(i, j) (i + (j * w))
		int idx = FLATTEN(x, y);
		color_t cval = in[idx];
		glm::vec3 sum(0, 0, 0);
		float cum_w = 0.f;
		for (ATrousFilter filter{ x,y,step,desc }; filter; ++filter) {
			auto data = *filter;

			idx = FLATTEN(data.nx, data.ny);
			color_t ctmp = in[idx];

			sum += ctmp * data.weight;
			cum_w += data.weight;
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

		if (desc.use_diffuse) {
			// divide by diffuse map, i.e. demodulate
			thrust::transform(
				thrust::device,
				bufs[buf_idx],
				bufs[buf_idx] + gbuf.size(),
				gbuf.get_diffuse(),
				bufs[buf_idx],
				Demodulate());
		}

		const dim3 block_size(8, 8);
		const dim3 blocks_per_grid(
			DIV_UP(desc.res[0], block_size.x),
			DIV_UP(desc.res[1], block_size.y));

		switch (desc.type) {
		case FilterType::ATROUS:
			for (int step = 1; step <= desc.filter_size; step <<= 1, desc.c_phi /= 2) {
#ifdef DENOISE_SHARED_MEM
				const dim3 block_size(4, 8);
				const dim3 blocks_per_grid(
					DIV_UP(desc.res[0], block_size.x),
					DIV_UP(desc.res[1], block_size.y));

				kern_atrous_denoise_shared<4,8> KERN_PARAM(blocks_per_grid, block_size) (bufs[1 - buf_idx], bufs[buf_idx], gbuf, step, desc);
#else
				kern_denoise<ATrousFilter> KERN_PARAM(blocks_per_grid, block_size) (bufs[1 - buf_idx], bufs[buf_idx], gbuf, step, desc);			
#endif

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
		case FilterType::BLUR:
			for (int step = 1; step <= desc.filter_size; step <<= 1, desc.c_phi /= 2) {
				kern_denoise_pure_blur KERN_PARAM(blocks_per_grid, block_size) (bufs[1 - buf_idx], bufs[buf_idx], step, desc);
				checkCUDAError("denoise");
				buf_idx = 1 - buf_idx;
			}
		default:
			break;
		}

		if (desc.use_diffuse) {
			// multiply by diffuse map, i.e. modulate
			thrust::transform(
				thrust::device,
				bufs[buf_idx],
				bufs[buf_idx] + gbuf.size(),
				gbuf.get_diffuse(),
				bufs[buf_idx],
				Modulate());
		}

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