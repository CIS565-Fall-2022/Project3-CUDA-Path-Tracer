#pragma once
#include <glm/glm.hpp>
#include <cuda.h>
#include <vector_types.h>
#include "../utilities.h"
#include "../sceneStructs.h"

namespace Denoiser {
	struct ParamDesc {
		ParamDesc() {}
		ParamDesc(int filter_size, glm::ivec2 res, float c_phi, float n_phi, float p_phi)
			: filter_size(filter_size), res(res), c_phi(c_phi), n_phi(n_phi), p_phi(p_phi) {
			// set up kernel and offset
			for (int i = -2, k = 0; i <= 2; ++i) {
				for (int j = -2; j <= 2; ++j) {
					offsets[k] = glm::ivec2(i, j);
					int x = glm::min(glm::abs(i), glm::abs(j));
					if (x == 0) {
						kernel[k] = 3. / 8;
					} else if (x == 1) {
						kernel[k] = 1. / 4;
					} else {
						kernel[k] = 1. / 16;
					}
					++k;
				}
			}
		}

		int filter_size;
		glm::ivec2 offsets[25];
		float kernel[25];
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