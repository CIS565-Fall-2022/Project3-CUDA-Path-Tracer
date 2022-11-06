#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <limits>
#include <functional>
#include <cuda.h>
#include "../utilities.h"
#include "../Collision/AABB.h"
#include "../scene.h"
#include "../intersections.cuh"

typedef size_t node_id_t;
static constexpr node_id_t null_id = 0;
static constexpr node_id_t root_id = 1;

struct nodeGPU;
struct octreeGPU;

struct leaf_data {
	leaf_data(int triangle_id, int geom_id)
		: triangle_id(triangle_id), geom_id(geom_id) {}
	int triangle_id; // -1 if the geom is a primitive
	int geom_id; // geom that this triangle belongs to
};
struct node {
	AABB bounds;
	node_id_t children[8];
	std::vector<leaf_data> leaf_infos;
	node(nodeGPU const& nodeGPU);
	node(AABB const& bounds) : bounds(bounds) {
		for (size_t i = 0; i < 8; ++i) {
			children[i] = null_id;
		}
	}
	bool is_leaf() const {
		return leaf_infos.size() == 0;
	}
};
struct nodeGPU {
	AABB bounds;
	node_id_t children[8];
	Span<leaf_data> leaf_infos;

	__host__ nodeGPU() {
		for (size_t i = 0; i < 8; ++i) {
			children[i] = null_id;
		}
	}
	__host__ nodeGPU(node const& o) : bounds(o.bounds) {
		for (size_t i = 0; i < 8; ++i) {
			children[i] = o.children[i];
		}
		leaf_infos = make_span(o.leaf_infos);
	}
	__host__ __device__ bool is_leaf() const {
		return leaf_infos.size() != 0;
	}
};

class octree {
	friend class octreeGPU;
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

		for (int geom_id = 0; geom_id < geoms.size(); ++geom_id) {
			auto const& geom = geoms[geom_id];
			if (!AABBIntersect(geom.bounds, box)) {
				continue;
			}
			if (geom.type != MESH) {
#ifdef OCTREE_MESH_ONLY
				continue;
#endif // OCTREE_MESH_ONLY

				if (leaf == null_id) {
					return true;
				} else {
					_nodes[leaf].leaf_infos.emplace_back(-1, geom_id);
				}
			} else {
				for (int i = meshes[geom.meshid].tri_start; i < meshes[geom.meshid].tri_end; ++i) {
					auto const& tri = tris[i];

					glm::vec3 triangle_verts[3];
					for (int x = 0; x < 3; ++x) {
						triangle_verts[x] = glm::vec3(geom.transform * glm::vec4(verts[tri.verts[x]], 1));
					}
					if (AABBTriangleIntersect(box, triangle_verts)) {
						if (leaf == null_id) {
							return true;
						} else {
							_nodes[leaf].leaf_infos.emplace_back(i, geom_id);
						}
					}
				}
			}
		}

		if (leaf == null_id) {
			return false;
		} else {
			return !_nodes[leaf].leaf_infos.empty();
		}
	}
	void build(Scene const& scene, node_id_t const cur, int const depth) {
		if (depth > _depth_lim) {
			return;
		} else if (depth == _depth_lim) {
			// build leaf
			get_hits(scene, _nodes[cur].bounds, cur);
			// put prims before meshes
			std::partition(_nodes[cur].leaf_infos.begin(), _nodes[cur].leaf_infos.end(), [](leaf_data const& data) {
				return data.triangle_id == -1; });
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
			bs[i] = AABB(mins[i] - OCTREE_BOX_EPS, mins[i] + half_size + OCTREE_BOX_EPS);
			if (get_hits(scene, bs[i], null_id)) {
				node_id_t ret = new_node(bs[i]);
				_nodes[cur].children[i] = ret;

				//_nodes[cur].children[i] = new_node(bs[i]);
				// TODO: figure out why _nodes[cur].children[i] = new_node(bs[i]) is wrong in release with /O2 flag
				// my mind is BLOWN by this fact

				build(scene, _nodes[cur].children[i], depth + 1);
			}
		}
	}
public:
	octree(octree const&) = delete;
	octree(octree&&) = delete;
	octree(Scene const& scene, AABB const& root_aabb, int depth_lim) : _depth_lim(depth_lim) {
		new_node(AABB()); //dummy node
		new_node(root_aabb); // root
		build(scene, root_id, 0);
	}

	octree(octreeGPU const& treeGPU);

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
	Span<nodeGPU> _nodes;
	MeshInfo _mesh_info;
	Span<Geom> _geoms;
	bool _is_copy;

	octreeGPU(octreeGPU const& o) 
		: _nodes(o._nodes), _mesh_info(o._mesh_info), _geoms(o._geoms), _is_copy(true) { }
	octreeGPU(octreeGPU&&) = delete;
	octreeGPU(octree const& tree, MeshInfo mesh_info, Span<Geom> geoms) : _is_copy(false) {
		// save mesh info of the scene
		this->_mesh_info = mesh_info;
		this->_geoms = geoms;

		int num_nodes = tree._nodes.size();
		// first form GPU representation at host side
		std::vector<nodeGPU> hst_nodes(num_nodes);
		for (size_t i = 0; i < num_nodes; ++i) {
			hst_nodes[i] = nodeGPU(tree._nodes[i]);
		}

		// upload to GPU
		nodeGPU* tmp;
		ALLOC(tmp, num_nodes);
		H2D(tmp, hst_nodes.data(), num_nodes);
		_nodes = Span<nodeGPU>(num_nodes, tmp);
	}
	~octreeGPU() {
		if (_is_copy) {
			return;
		}

		size_t num_nodes = _nodes.size();
		// get data back from GPU so we can free it
		nodeGPU* hst_nodes = new nodeGPU[num_nodes];
		D2H(hst_nodes, _nodes, num_nodes);
		for (size_t i = 0; i < num_nodes; ++i) {
			FREE(hst_nodes[i].leaf_infos);
		}
		FREE(_nodes);
		delete[] hst_nodes;
	}

	__device__ bool handle_leaf(node_id_t const cur, ShadeableIntersection& inters, Ray const& ray) const {
		auto const& info = _nodes[cur].leaf_infos;
		float t_min = inters.t;

		// any closer hit?
		bool any_hit = false;

		// loop through primitives
		int i;
		for (i = 0; i < info.size() && info[i].triangle_id == -1; ++i) {
			auto const& geom = _geoms[info[i].geom_id];

			ShadeableIntersection tmp_inters;
			if (geom.type == CUBE) {
				float t = boxIntersectionTest(geom, ray, tmp_inters);
				if (t > 0) {
					if (t_min > t) {
						t_min = t;
						inters = tmp_inters;
						any_hit = true;
					}
				}
			} else if (geom.type == SPHERE) {
				float t = sphereIntersectionTest(geom, ray, tmp_inters);
				if (t > 0) {
					if (t_min > t) {
						t_min = t;
						inters = tmp_inters;
						any_hit = true;
					}
				}
			} else {
				return false;
			}
		}

		// loop through triangles
		int idx = -1;
		glm::vec3 barycoord;
		float t_min_tri = FLT_MAX; // use another var because t is apparently different in local space???
		for (; i < info.size(); ++i) {
			auto const& tri = _mesh_info.tris[info[i].triangle_id];
			auto const& geom = _geoms[info[i].geom_id];
			glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(ray.origin, 1.0f));
			glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(ray.direction, 0.0f)));
			glm::vec3 tmp_barycoord;
			glm::vec3 triangle_verts[3]{
				_mesh_info.vertices[tri.verts[0]],
				_mesh_info.vertices[tri.verts[1]],
				_mesh_info.vertices[tri.verts[2]]
			};

			if (glm::intersectRayTriangle(ro, rd, triangle_verts[0], triangle_verts[1], triangle_verts[2], tmp_barycoord)) {
				float t = tmp_barycoord.z;
				if (t_min_tri > t) {
					t_min_tri = t;
					idx = i;
					barycoord = tmp_barycoord;
				}
			}
		}

		if (idx != -1) {
			// result is a ray-triangle intersection
			ShadeableIntersection tmp_inters;
			float t = intersFromTriangle(
				tmp_inters,
				ray,
				t_min_tri,
				_mesh_info,
				_geoms[info[idx].geom_id],
				_mesh_info.tris[info[idx].triangle_id],
				glm::vec2(barycoord));

			if (t_min > t) {
				t_min = t;
				inters = tmp_inters;
				any_hit = true;
			}
		}
		return any_hit;
	}

	/// <summary>
	/// recursively searches the octree for hits
	/// </summary>
	/// <param name="cur"> current node </param>
	/// <param name="info"> hit info container </param>
	/// <param name="ray"> ray </param>
	/// <returns>whether there are any triangle hits </returns>
	__device__ bool _search(node_id_t const cur, ShadeableIntersection& inters, Ray const& ray) const {
		// FOR THE LOVE OF GOD
		// Why can't recursion just work on GPU ????
		/*
		if (_nodes[cur].is_leaf()) {
			return handle_leaf(cur, inters, ray);
		} else {
			for (size_t i = 0; i < 8; ++i) {
				node_id_t child = _nodes[cur].children[i];
				if (child != null_id && AABBRayIntersect(_nodes[child].bounds, ray, nullptr)) {
					if (_search(child, inters, ray)) {
						return true;
					}
				}
			}
			return false;
		}*/

		bool ret = false;
		node_id_t stack[10 * OCTREE_DEPTH];
		node_id_t* pstk = stack;
		for (size_t i = 0; i < 8; ++i) {
			node_id_t child = _nodes[root_id].children[i];
			if (child != null_id && AABBRayIntersect(_nodes[child].bounds, ray, nullptr)) {
				*(pstk++) = child;
			}
		}
		while (pstk != stack) {
			node_id_t id = *(--pstk);
			if (_nodes[id].is_leaf()) {
				ret |= handle_leaf(id, inters, ray);
			} else {
				for (size_t i = 0; i < 8; ++i) {
					node_id_t child = _nodes[id].children[i];
					if (child != null_id && AABBRayIntersect(_nodes[child].bounds, ray, nullptr)) {
						*(pstk++) = child;
					}
				}
			}
		}
		return ret;
	}
	__device__ bool search(ShadeableIntersection& inters, Ray const& ray) const {
		inters.t = FLT_MAX;
		bool any_hit = false;
#ifdef OCTREE_MESH_ONLY
		for (int i = 0; i < _geoms.size(); i++) {
			Geom const& geom = _geoms[i];
#ifdef AABB_CULLING
			if (!AABBRayIntersect(geom.bounds, ray, nullptr))
				continue;
#endif // AABB_CULLING

			float t;
			ShadeableIntersection tmp;

			if (geom.type == CUBE) {
				t = boxIntersectionTest(geom, ray, tmp);
			} else if (geom.type == SPHERE) {
				t = sphereIntersectionTest(geom, ray, tmp);
			} else {
				continue;
			}
			if (t > 0.0f && inters.t > t) {
				inters = tmp;
				inters.t = t;
				any_hit = true;
			}
		}
#endif // OCTREE_MESH_ONLY
		if (_search(root_id, inters, ray)) {
			any_hit = true;
		}
		return any_hit;
	}
};

inline octree::octree(octreeGPU const& treeGPU) : _depth_lim(OCTREE_DEPTH) {
#pragma warning(push)
#pragma warning(push)
#pragma warning(disable : 6385)
#pragma warning(disable : 6011)

	size_t num_nodes = treeGPU._nodes.size();
	nodeGPU* tmp = reinterpret_cast<nodeGPU*>(malloc(num_nodes * sizeof(nodeGPU)));
	D2H(tmp, treeGPU._nodes, num_nodes);
	for (int i = 0; i < num_nodes; ++i) {
		_nodes.emplace_back(tmp[i]);
	}
	free(tmp);

#pragma warning(pop)
#pragma warning(pop)
}

inline node::node(nodeGPU const& nodeGPU)
	: bounds(nodeGPU.bounds) {
#pragma warning(push)
#pragma warning(push)
#pragma warning(disable : 6385)
#pragma warning(disable : 6011)

	for (size_t i = 0; i < 8; ++i) {
		children[i] = nodeGPU.children[i];
	}

	size_t num_data = nodeGPU.leaf_infos.size();
	leaf_data* tmp = reinterpret_cast<leaf_data*>(malloc(num_data * sizeof(leaf_data)));
	D2H(tmp, nodeGPU.leaf_infos, num_data);
	
	for (int i = 0; i < num_data; ++i) {
		leaf_infos.emplace_back(tmp[i].triangle_id, tmp[i].geom_id);
	}
	free(tmp);

#pragma warning(pop)
#pragma warning(pop)
}

