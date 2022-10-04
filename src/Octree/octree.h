#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <limits>
#include <functional>
#include <cuda.h>
#include "../utilities.h"
#include "../Collision/AABB.h"
#include "../scene.h"
#include "../intersections.h"

/// <summary>
/// octree for triangles
/// primitives can be culled using simple AABB
/// </summary>
class octree {
	friend class octreeGPU;

public:
	typedef size_t node_id_t;
	static constexpr node_id_t null_id = 0;
	static constexpr node_id_t root_id = 1;
	static constexpr float eps_scale = 100.0f;
	static constexpr float eps = eps_scale * std::numeric_limits<float>::epsilon();

	struct node {
		AABB bounds;
		node_id_t children[8];
		std::vector<int> triangles;

		node(AABB const& bounds) : bounds(bounds) {
			for (size_t i = 0; i < 8; ++i) {
				children[i] = null_id;
			}
		}
		bool is_leaf() const {
			for (node_id_t id : children) {
				if (id != null_id) {
					return false;
				}
			}
			return true;
		}
	};

private:
	std::vector<node> _nodes;
	int _depth_lim;


	node_id_t new_node(AABB const& bounds) {
		node_id_t ret = _nodes.size();
		_nodes.emplace_back(bounds);
		return ret;
	}
	// does any triangles in the scene hit the AABB?
	// if leaf is not null, fills it with the intersection info
	bool get_hits(Scene const& scene, AABB const& box, node_id_t leaf) {
		auto const& meshes = scene.meshes;
		auto const& verts = scene.vertices;
		auto const& tris = scene.triangles;
		auto const& geoms = scene.geoms;

		for (auto const& geom : geoms) {
			if (geom.type != MESH) {
				continue;
			}
			if (!intersect(geom.bounds, box)) {
				continue;
			}

			for (int i = meshes[geom.meshid].tri_start; i < meshes[geom.meshid].tri_end; ++i) {
				auto const& tri = tris[i];

				glm::vec3 triangle_verts[3];
				for (int x = 0; x < 3; ++x) {
					triangle_verts[x] = glm::vec3(geom.transform * glm::vec4(verts[tri.verts[x]], 1));
				}
				if (intersect(box, triangle_verts)) {
					if (leaf == null_id) {
						return true;
					} else {
						_nodes[leaf].triangles.push_back(i);
					}
				}
			}
		}

		if (leaf == null_id) {
			return false;
		} else {
			return !_nodes[leaf].triangles.empty();
		}
	}
	void build(Scene const& scene, node_id_t const cur, int const depth) {
		if (depth > _depth_lim) {
			return;
		} else if (depth == _depth_lim) {
			// build leaf
			get_hits(scene, _nodes[cur].bounds, cur);
			return;
		}

		// recursively divide the space
		glm::vec3 half_size = _nodes[cur].bounds.extent();
		glm::vec3 half_X = glm::vec3(half_size.x, 0, 0);
		glm::vec3 half_Y = glm::vec3(0, half_size.y, 0);
		glm::vec3 half_Z = glm::vec3(0, 0, half_size.z);
		glm::vec3 bmin = _nodes[cur].bounds.min();
		AABB bs[8];
		glm::vec3 mins[8]{
			bmin,
			bmin + half_Z,
			bmin + half_Y,
			bmin + half_Y + half_Z,
			bmin + half_X,
			bmin + half_X + half_Z,
			bmin + half_X + half_Y,
			bmin + half_X + half_Y + half_Z,
		};

		for (size_t i = 0; i < 8; ++i) {
			bs[i] = AABB(mins[i] - eps, mins[i] + half_size + eps);
			if (get_hits(scene, bs[i], null_id)) {
				node_id_t ret = new_node(bs[i]);
				_nodes[cur].children[i] = ret;

//				_nodes[cur].children[i] = new_node(bs[i]);
				// TODO: figure out why _nodes[cur].children[i] = new_node(bs[i]) is wrong in release

				build(scene, _nodes[cur].children[i], depth + 1);
			}
		}
	}
public:
	octree(Scene const& scene, AABB const& root_aabb, int depth_lim) : _depth_lim(depth_lim) {
		new_node(AABB()); //dummy node
		new_node(root_aabb); // root
		build(scene, root_id, 0);
	}

	template<typename Callback>
	void dfs(Callback func) {
		std::function<void(node_id_t, int)> f = [&](node_id_t cur, int depth) {
			func(_nodes[cur], depth);
			for (node_id_t child : _nodes[cur].children) {
				if (child != null_id) {
					f(child, depth + 1);
				}
			}
		};
		f(root_id, 0);
	}
};

/// <summary>
/// GPU side representation of the octree 
/// which cannot be modified
/// </summary>
struct octreeGPU {
	struct node {
		AABB bounds;
		octree::node_id_t children[8];
		int* triangles;
		node(AABB const& bounds) : bounds(bounds) {
			for (size_t i = 0; i < 8; ++i) {
				children[i] = octree::null_id;
			}
		}
		bool is_leaf() const {
			for (octree::node_id_t id : children) {
				if (id != octree::null_id) {
					return false;
				}
			}
			return true;
		}
	};

	Span<node> nodes;
	__host__ octreeGPU() { }
	__host__ void from(octree const& tree) {
		//int num_nodes;
		//ALLOC(nodes, )
	}
};