#define TINYGLTF_IMPLEMENTATION
#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tiny_gltf.h"
#include "stb_image.h"


Scene::Scene(string filename) {
	cout << "Reading scene from " << filename << " ..." << endl;
	cout << " " << endl;
	char* fname = (char*)filename.c_str();
	fp_in.open(fname);
	if (!fp_in.is_open()) {
		cout << "Error reading from file - aborting!" << endl;
		throw;
	}

	while (fp_in.good()) {
		string line;
		utilityCore::safeGetline(fp_in, line);
		if (!line.empty()) {
			vector<string> tokens = utilityCore::tokenizeString(line);
			if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
				loadMaterial(tokens[1]);
				cout << " " << endl;
			}
			else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
				loadGeom(tokens[1]);
				cout << " " << endl;
			}
			else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
				loadCamera();
				cout << " " << endl;
			}
		}
	}
}
//reference: http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/
glm::vec3 triangleTangent(Triangle& tri)
{
	glm::vec3 deltaPos1 = tri.pos[1] - tri.pos[0];
	glm::vec3 deltaPos2 = tri.pos[2] - tri.pos[0];
	glm::vec2 deltaUV1 = tri.uv[1] - tri.uv[0];
	glm::vec2 deltaUV2 = tri.uv[2] - tri.uv[0];
	glm::vec3 tangent = (deltaUV2.y * deltaPos1 - deltaUV1.y * deltaPos2) / (deltaPos2.y * deltaUV1.x - deltaUV1.y * deltaUV2.x);;

	return glm::normalize(tangent);
}

// Reference example code: https://github.com/syoyo/tinygltf
int Scene::loadGLTF(string path, Geom& geom) {
	tinygltf::Model model;
	tinygltf::TinyGLTF loader;
	std::string err;
	std::string warn;

	bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, path);
	//bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, argv[1]); //

	if (!warn.empty())
	{
		cout << "Warning: " << warn << endl;
	}
	if (!err.empty())
	{
		cout << "Error: " << err << endl;
	}
	if (!ret)
	{
		cout << "Failed to parse glTF" << endl;
		return -1;
	}

	geom.mesh_start_idx = triangles.size();
	for (auto& mesh : model.meshes)
	{
		for (auto& prim : mesh.primitives)
		{
			const tinygltf::Accessor* accessor;
			const tinygltf::BufferView* bufferView;
			const tinygltf::Buffer* buffer;
			int offset;
			const unsigned short* indices = nullptr;
			const float* positions = nullptr;
			const float* normals = nullptr;
			const float* uvs = nullptr;
			const float* tangents = nullptr;

			accessor = &model.accessors[prim.indices];
			bufferView = &model.bufferViews[accessor->bufferView];
			buffer = &model.buffers[bufferView->buffer];
			offset = bufferView->byteOffset + accessor->byteOffset;
			indices = reinterpret_cast<const unsigned short*>(&buffer->data[offset]);

			accessor = &model.accessors[prim.attributes["POSITION"]];
			bufferView = &model.bufferViews[accessor->bufferView];
			buffer = &model.buffers[bufferView->buffer];
			offset = bufferView->byteOffset + accessor->byteOffset;
			positions = reinterpret_cast<const float*>(&buffer->data[offset]);

			if (prim.attributes["NORMAL"] >= 0)
			{
				accessor = &model.accessors[prim.attributes["NORMAL"]];
				bufferView = &model.bufferViews[accessor->bufferView];
				buffer = &model.buffers[bufferView->buffer];
				offset = bufferView->byteOffset + accessor->byteOffset;
				normals = reinterpret_cast<const float*>(&buffer->data[offset]);
			}

			if (prim.attributes["TEXCOORD_0"] >= 0)
			{
				accessor = &model.accessors[prim.attributes["TEXCOORD_0"]];
				bufferView = &model.bufferViews[accessor->bufferView];
				buffer = &model.buffers[bufferView->buffer];
				offset = bufferView->byteOffset + accessor->byteOffset;
				uvs = reinterpret_cast<const float*>(&buffer->data[offset]);
			}

			if (prim.attributes["TANGENT"] >= 0)
			{
				accessor = &model.accessors[prim.attributes["TANGENT"]];
				bufferView = &model.bufferViews[accessor->bufferView];
				buffer = &model.buffers[bufferView->buffer];
				offset = bufferView->byteOffset + accessor->byteOffset;
				tangents = reinterpret_cast<const float*>(&buffer->data[offset]);
			}

			for (size_t i = 0; i < model.accessors[prim.indices].count; i += 3)
			{
				Triangle tri;
				for (int j = 0; j < 3; ++j)
				{
					int idx = indices[i + j];
					tri.pos[j] = glm::vec3(positions[idx * 3], positions[idx * 3 + 1], positions[idx * 3 + 2]);
					if (normals)
					{
						tri.normal[j] = glm::vec3(normals[idx * 3], normals[idx * 3 + 1], normals[idx * 3 + 2]);
					}
					if (uvs)
					{
						tri.uv[j] = glm::vec2(uvs[idx * 2], uvs[idx * 2 + 1]);
					}
					if (tangents)
					{
						tri.tangent[j] = glm::vec4(tangents[idx * 4], tangents[idx * 4 + 1], tangents[idx * 4 + 2], tangents[idx * 4 + 3]);
					}
					glm::vec3 worldPos(geom.transform * glm::vec4(tri.pos[j], 1.f));
				}
				if (!normals)
				{
					glm::vec3 normal = glm::normalize(glm::cross(tri.pos[1] - tri.pos[0], tri.pos[2] - tri.pos[0]));
					for (int i = 0; i < 3; ++i)
					{
						tri.normal[i] = normal;
					}
				}
				if (uvs && !tangents)
				{
					glm::vec3 t = triangleTangent(tri);
					for (int i = 0; i < 3; ++i)
					{
						tri.tangent[i] = glm::vec4(t, 1);
					}
				}
				triangles.push_back(tri);
			}
		}
	}
	geom.mesh_end_idx = triangles.size();
	return 1;
}

int Scene::loadGeom(string objectid) {
	int id = atoi(objectid.c_str());
	if (id != geoms.size()) {
		cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
		return -1;
	}
	else {
		cout << "Loading Geom " << id << "..." << endl;
		Geom newGeom;
		string line;

		//load object type
		utilityCore::safeGetline(fp_in, line);
		if (!line.empty() && fp_in.good()) {
			if (strcmp(line.c_str(), "sphere") == 0) {
				cout << "Creating new sphere..." << endl;
				newGeom.type = SPHERE;
			}
			else if (strcmp(line.c_str(), "cube") == 0) {
				cout << "Creating new cube..." << endl;
				newGeom.type = CUBE;
			}
			else if (strcmp(line.c_str(), "gltf") == 0) {
				cout << "Creating new MESH..." << endl;
				newGeom.type = MESH;
				utilityCore::safeGetline(fp_in, line);
				loadGLTF(line, newGeom);
			}
		}

		//link material
		utilityCore::safeGetline(fp_in, line);
		if (!line.empty() && fp_in.good()) {
			vector<string> tokens = utilityCore::tokenizeString(line);
			newGeom.materialid = atoi(tokens[1].c_str());
			cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
		}

		//load transformations
		utilityCore::safeGetline(fp_in, line);
		while (!line.empty() && fp_in.good()) {
			vector<string> tokens = utilityCore::tokenizeString(line);

			//load tranformations
			if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
				newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
				newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
				newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}

			utilityCore::safeGetline(fp_in, line);
		}

		newGeom.transform = utilityCore::buildTransformationMatrix(
			newGeom.translation, newGeom.rotation, newGeom.scale);
		newGeom.inverseTransform = glm::inverse(newGeom.transform);
		newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

		geoms.push_back(newGeom);
		return 1;
	}
}

int Scene::loadCamera() {
	cout << "Loading Camera ..." << endl;
	RenderState& state = this->state;
	Camera& camera = state.camera;
	float fovy;

	//load static properties
	for (int i = 0; i < 5; i++) {
		string line;
		utilityCore::safeGetline(fp_in, line);
		vector<string> tokens = utilityCore::tokenizeString(line);
		if (strcmp(tokens[0].c_str(), "RES") == 0) {
			camera.resolution.x = atoi(tokens[1].c_str());
			camera.resolution.y = atoi(tokens[2].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
			fovy = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
			state.iterations = atoi(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
			state.traceDepth = atoi(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
			state.imageName = tokens[1];
		}
	}

	string line;
	utilityCore::safeGetline(fp_in, line);
	while (!line.empty() && fp_in.good()) {
		vector<string> tokens = utilityCore::tokenizeString(line);
		if (strcmp(tokens[0].c_str(), "EYE") == 0) {
			camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
			camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "UP") == 0) {
			camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "FOCAL") == 0) {
			camera.focalLength = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "APERTURE") == 0) {
			camera.aperture = atof(tokens[1].c_str());
		}
		utilityCore::safeGetline(fp_in, line);
	}

	//calculate fov based on resolution
	float yscaled = tan(fovy * (PI / 180));
	float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
	float fovx = (atan(xscaled) * 180) / PI;
	camera.fov = glm::vec2(fovx, fovy);

	camera.right = glm::normalize(glm::cross(camera.view, camera.up));
	camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
		2 * yscaled / (float)camera.resolution.y);

	camera.view = glm::normalize(camera.lookAt - camera.position);

	//set up render camera stuff
	int arraylen = camera.resolution.x * camera.resolution.y;
	state.image.resize(arraylen);
	std::fill(state.image.begin(), state.image.end(), glm::vec3());

	cout << "Loaded camera!" << endl;
	return 1;
}

int Scene::loadMaterial(string materialid) {
	int id = atoi(materialid.c_str());
	if (id != materials.size()) {
		cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
		return -1;
	}
	else {
		cout << "Loading Material " << id << "..." << endl;
		Material newMaterial;

		//load static properties
		for (int i = 0; i < 9; i++) {
			string line;
			Map* map = nullptr;
			utilityCore::safeGetline(fp_in, line);
			vector<string> tokens = utilityCore::tokenizeString(line);
			if (strcmp(tokens[0].c_str(), "RGB") == 0) {
				glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				newMaterial.color = color;
			}
			else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
				newMaterial.specular.exponent = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
				glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				newMaterial.specular.color = specColor;
			}
			else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
				newMaterial.hasReflective = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
				newMaterial.hasRefractive = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
				newMaterial.indexOfRefraction = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
				newMaterial.emittance = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "TEXTUREMAP") == 0) {
				if (strcmp(tokens[1].c_str(), "NULL") != 0) {
					map = &newMaterial.texture_map;
					int width, height, channels;
					unsigned char* pixels = stbi_load(tokens[1].c_str(), &width, &height, &channels, 3);
					map->offset = maps.size();
					map->dim = glm::ivec2(width, height);
					for (int i = 0; i < width * height; i++) {
						glm::vec3 col;
						for (int j = 0; j < 3; ++j) {
							col[j] = (float)pixels[i * 3 + j] / 255.f;
						}
						maps.push_back(col);
					}
					stbi_image_free(pixels);
				}
			}
			else if (strcmp(tokens[0].c_str(), "NORMALMAP") == 0) {
				if (strcmp(tokens[1].c_str(), "NULL") != 0) {
					map = &newMaterial.normal_map;
					int width, height, channels;
					unsigned char* pixels = stbi_load(tokens[1].c_str(), &width, &height, &channels, 3);
					map->offset = maps.size();
					map->dim = glm::ivec2(width, height);
					for (int i = 0; i < width * height; i++) {
						glm::vec3 col;
						for (int j = 0; j < 3; ++j) {
							col[j] = (float)pixels[i * 3 + j] / 255.f;
						}
						maps.push_back(col);
					}
					stbi_image_free(pixels);
				}
			}
		}
		materials.push_back(newMaterial);
		return 1;
	}
}
