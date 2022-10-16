#pragma once
#include <glm/glm.hpp>
#include <cuda.h>
#include <vector_types.h>
#include "../utilities.h"
#include "../sceneStructs.h"

// functors
struct RadianceToNormalizedRGB {
	int iter;
	RadianceToNormalizedRGB(int iter) : iter(iter) { }
	__device__ color_t operator()(color_t const& r) const {
		if (!iter) {
			return color_t(0);
		}
		return glm::clamp(r / (float)iter, 0.f, 1.f);
	}
};
struct RadianceToRGBA {
	int iter;
	RadianceToRGBA(int iter) : iter(iter) { }
	__device__ uchar4 operator()(color_t const& r) const {
		if (!iter) {
			return make_uchar4(0, 0, 0, 0);
		}
		return make_uchar4(
			glm::clamp((int)(r.x / iter * 255.f), 0, 255),
			glm::clamp((int)(r.y / iter * 255.f), 0, 255),
			glm::clamp((int)(r.z / iter * 255.f), 0, 255),
			0);
	}
};
struct NormalizedRGBToRGBA {
	__device__ uchar4 operator()(color_t const& r) const {
		return make_uchar4(
			glm::clamp((int)(r.x * 255.f), 0, 255),
			glm::clamp((int)(r.y * 255.f), 0, 255),
			glm::clamp((int)(r.z * 255.f), 0, 255),
			0);
	}
};

struct PosToRGBA {
	glm::vec3 min_pos, max_pos;
	PosToRGBA(glm::vec3 min_pos, glm::vec3 max_pos) 
		: min_pos(min_pos), max_pos(max_pos) { }
	__device__ uchar4 operator()(glm::vec3 pos) const {
		pos = (pos - min_pos) / (max_pos - min_pos);

		return make_uchar4(
			glm::clamp((int)(pos.x * 255.f), 0, 255),
			glm::clamp((int)(pos.y * 255.f), 0, 255),
			glm::clamp((int)(pos.z * 255.f), 0, 255),
			0);
	}
};

struct NormalToRGBA {
	__device__ uchar4 operator()(glm::vec3 const& n) const {
		// convert from the range [-1, 1] to [0, 1]
		return make_uchar4(
			glm::clamp((int)(((n.x + 1.f) / 2.f) * 255.f), 0, 255),
			glm::clamp((int)(((n.y + 1.f) / 2.f) * 255.f), 0, 255),
			glm::clamp((int)(((n.z + 1.f) / 2.f) * 255.f), 0, 255),
			0);
	}
};

namespace Denoiser {
	enum FilterType {
		ATROUS,
		GAUSSIAN,
		BLUR,
		NUM_FILTERS
	};

	struct ParamDesc {
		ParamDesc(FilterType type, int filter_size, glm::ivec2 res, float c_phi, float n_phi, float p_phi)
			: use_diffuse(true), type(type), filter_size(filter_size), s_dev(7), res(res), c_phi(c_phi), n_phi(n_phi), p_phi(p_phi) { }

		bool use_diffuse;
		FilterType type;
		int filter_size;
		float s_dev; //standard deviation, only used by Gaussian
		glm::ivec2 res;
		float c_phi, n_phi, p_phi;
	};

	// functors
	struct IntersectionToNormal {
		__host__ __device__ glm::vec3 operator()(ShadeableIntersection const& s) const {
			return s.surfaceNormal;
		}
	};
	struct IntersectionToPos {
		__host__ __device__ glm::vec3 operator()(ShadeableIntersection const& s) const {
			if (s.t < 0.f) {
				return glm::vec3(0);
			}
			return s.hitPoint;
		}
	};
	struct IntersectionToDiffuse {
		Material const* mats;
		IntersectionToDiffuse(Material const* mats) : mats(mats) { }
		__host__ __device__ color_t operator()(ShadeableIntersection const& s) const {
			if (s.t < 0.f) {
				return BACKGROUND_COLOR;
			}
			if (mats[s.materialId].textures.diffuse != -1) {
				return s.tex_color;
			} else {
				return mats[s.materialId].diffuse;
			}
		}
	};
}