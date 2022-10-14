#pragma once
#include <glm/glm.hpp>
#include <cuda.h>
#include <vector_types.h>
#include "../utilities.h"
#include "../sceneStructs.h"

namespace Denoiser {
	enum FilterType {
		ATROUS,
		GAUSSIAN,
		NUM_FILTERS
	};

	struct ParamDesc {
		ParamDesc(FilterType type, int filter_size, glm::ivec2 res, float c_phi, float n_phi, float p_phi)
			: type(type), filter_size(filter_size), s_dev(1), res(res), c_phi(c_phi), n_phi(n_phi), p_phi(p_phi) { }

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
			return s.hitPoint;
		}
	};
	struct IntersectionToDiffuse {
		Material const* mats;
		IntersectionToDiffuse(Material const* mats) : mats(mats) { }
		__host__ __device__ color_t operator()(ShadeableIntersection const& s) const {
			if (mats[s.materialId].textures.diffuse != -1) {
				return s.tex_color;
			} else {
				return mats[s.materialId].diffuse;
			}
		}
	};
}