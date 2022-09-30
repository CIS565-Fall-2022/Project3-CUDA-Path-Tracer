#include "main.h"
#include "utilities.h"
#include <fstream>
#include <glm/glm.hpp>

#define SEP " "

// I don't have time to refactor the camera code, so here we are ...
extern JunksFromMain g_mainJunks;

template<typename T, glm::precision P>
std::ostream& operator<< (std::ostream& out, glm::tvec2<T,P> const& v) {
	out << v.x << SEP << v.y;
	return out;
}
template<typename T, glm::precision P>
std::ostream& operator<< (std::ostream& out, glm::tvec3<T,P> const& v) {
	out << v.x << SEP << v.y << SEP << v.z;
	return out;
}
template<typename T>
std::ostream& operator<< (std::ostream& out, std::vector<T> const& v) {
	out << v.size() << SEP;
	for (size_t i = 0; i < v.size(); ++i) {
		out << v[i] << SEP;
	}
	return out;
}
template<typename T, glm::precision P>
std::istream& operator>> (std::istream& in, glm::tvec2<T,P>& v) {
	in >> v.x >> v.y;
	return in;
}
template<typename T, glm::precision P>
std::istream& operator>> (std::istream& in, glm::tvec3<T,P>& v) {
	in >> v.x >> v.y >> v.z;
	return in;
}
template<typename T>
std::istream& operator>> (std::istream& in, std::vector<T>& v) {
	size_t sz;
	in >> sz;
	v.resize(sz);
	for (size_t i = 0; i < v.size(); ++i) {
		in >> v[i];
	}
	return in;
}

bool save_state(int iter, RenderState const& state, Scene const& scene, char const* filename) {
	std::ofstream fout(save_files_dir + std::string(filename) + ".sav");
	if (!fout) {
		return false;
	}

	fout << g_mainJunks.phi << SEP;
	fout << g_mainJunks.theta << SEP;
	fout << g_mainJunks.zoom << SEP;
	fout << g_mainJunks.cameraPosition << SEP;
	fout << g_mainJunks.ogLookAt << SEP;

	fout << iter << SEP;
	fout << scene.filename << SEP;
	fout << state.camera.fov << SEP;
	fout << state.camera.lookAt << SEP;
	fout << state.camera.pixelLength << SEP;
	fout << state.camera.position << SEP;
	fout << state.camera.resolution << SEP;
	fout << state.camera.right << SEP;
	fout << state.camera.up << SEP;
	fout << state.camera.view << SEP;
	fout << state.image << SEP;
	fout << state.imageName << SEP;
	fout << state.iterations << SEP;
	fout << state.iterations << SEP;
	fout << state.traceDepth << SEP;
	return true;
}

bool read_state(char const* filename, int& iter, Scene*& scene) {
	std::ifstream fin(filename);
	if (!fin) {
		return false;
	}

	fin >> g_mainJunks.phi;
	fin >> g_mainJunks.theta;
	fin >> g_mainJunks.zoom;
	fin >> g_mainJunks.cameraPosition;
	fin >> g_mainJunks.ogLookAt;

	fin >> iter;
	std::string scene_file;
	fin >> scene_file;
	scene = new Scene(scene_file, false);

	auto& state = scene->state;
	fin >> state.camera.fov;
	fin >> state.camera.lookAt;
	fin >> state.camera.pixelLength;
	fin >> state.camera.position;
	fin >> state.camera.resolution;
	fin >> state.camera.right;
	fin >> state.camera.up;
	fin >> state.camera.view;
	fin >> state.image;
	fin >> state.imageName;
	fin >> state.iterations;
	fin >> state.iterations;
	fin >> state.traceDepth;

	return true;
}