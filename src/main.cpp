#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "main.h"
#include "preview.h"
#include <cstring>
#include <sstream>
#include <fstream>
#include <iostream>
#include "utilities.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

#define LOADSTATE 0
//-------------------------------
//-------------MAIN--------------
//-------------------------------

bool addObj(Scene* scene, string filename) {
	std::string inputfile = filename;
	tinyobj::ObjReaderConfig reader_config;
	reader_config.mtl_search_path = "../materials"; // Path to material files

	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(inputfile, reader_config)) {
		if (!reader.Error().empty()) {
			std::cerr << "TinyObjReader: " << reader.Error();
		}
		return false;
	}

	if (!reader.Warning().empty()) {
		std::cout << "TinyObjReader: " << reader.Warning();
	}

	auto& attrib = reader.GetAttrib();
	auto& shapes = reader.GetShapes();
	auto& materials = reader.GetMaterials();

	int boundingIndex = scene->geoms.size();
	scene->geoms.push_back(Geom());
	scene->geoms[boundingIndex].type = BOUNDINGBOX;
	scene->geoms[boundingIndex].materialid = 1;

	int size = scene->faces.size();
	// Loop over shapes
	glm::vec3 min = glm::vec3(1e38, 1e38, 1e38);
	glm::vec3 max = glm::vec3(-1e38, -1e38, -1e38);

	for (size_t s = 0; s < shapes.size(); s++) {
		// Loop over faces(polygon)
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

			scene->faces.push_back(Geom());
			scene->faces[size].type = PLANE;
			scene->faces[size].materialid = 1;
			scene->faces[size].hasNorm = false;
			// Loop over vertices in the face.
			for (size_t v = 0; v < fv; v++) {
				// access to vertex
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
				tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
				tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
				tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
				if (vx > max.x) {
					max.x = vx;
				}
				if (vx < min.x) {
					min.x = vx;
				}
				if (vy > max.y) {
					max.y = vy;
				}
				if (vy < min.y) {
					min.y = vy;
				}
				if (vz > max.z) {
					max.z = vz;
				}
				if (vz < min.z) {
					min.z = vz;
				}

				if (v == 0) {
					scene->faces[size].p1 = glm::vec3(vx, vy, vz);
				}
				else if (v == 1) {
					scene->faces[size].p2 = glm::vec3(vx, vy, vz);
				}
				else if (v == 2) {
					scene->faces[size].p3 = glm::vec3(vx, vy, vz);
				}
				else {
					return false;
				}

				// Check if `normal_index` is zero or positive. negative = no normal data
				if (idx.normal_index >= 0) {
					tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
					tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
					tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
				}

				// Check if `texcoord_index` is zero or positive. negative = no texcoord data
				if (idx.texcoord_index >= 0) {
					tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
					tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
				}

				// Optional: vertex colors
				// tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
				// tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
				// tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
			}
			index_offset += fv;
			size++;

			// per-face material
			shapes[s].mesh.material_ids[f];
		}
	}

	max += glm::vec3(.01, .01, .01);
	min -= glm::vec3(.01, .01, .01);
	glm::vec3 center = (max + min) / 2.f;
	scene->geoms[boundingIndex].translation = center;
	scene->geoms[boundingIndex].rotation = glm::vec3(0,0,0);
	scene->geoms[boundingIndex].scale = max - min;
	scene->geoms[boundingIndex].transform = utilityCore::buildTransformationMatrix(
		scene->geoms[boundingIndex].translation, scene->geoms[boundingIndex].rotation, scene->geoms[boundingIndex].scale);
	scene->geoms[boundingIndex].inverseTransform = glm::inverse(scene->geoms[boundingIndex].transform);
	scene->geoms[boundingIndex].invTranspose = glm::inverseTranspose(scene->geoms[boundingIndex].transform);
	return true;
}

int main(int argc, char** argv) {
	startTimeString = currentTimeString();

	if (argc < 2) {
		printf("Usage: %s SCENEFILE.txt\n", argv[0]);
		return 1;
	}

	const char* sceneFile = argv[1];
	char* objFile;
	if (argc > 2) {
		objFile = argv[2];
	}

	// Load scene file
	scene = new Scene(sceneFile);
	//add obj
	if (argc > 2) {
		bool success = addObj(scene, objFile);
	}
	

	//Create Instance for ImGUIData
	guiData = new GuiDataContainer();

	// Set up camera stuff from loaded path tracer settings
	iteration = 0;
	renderState = &scene->state;
	Camera& cam = renderState->camera;
	renderState->usingSavedState = false;
	width = cam.resolution.x;
	height = cam.resolution.y;

	glm::vec3 view = cam.view;
	glm::vec3 up = cam.up;
	glm::vec3 right = glm::cross(view, up);
	up = glm::cross(right, view);

	cameraPosition = cam.position;

	// compute phi (horizontal) and theta (vertical) relative 3D axis
	// so, (0 0 1) is forward, (0 1 0) is up
	glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
	glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
	phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
	theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
	ogLookAt = cam.lookAt;
	zoom = glm::length(cam.position - ogLookAt);

	// Initialize CUDA and GL components
	init();

	// Initialize ImGui Data
	InitImguiData(guiData);
	InitDataContainer(guiData);

	// GLFW main loop
	mainLoop();

	return 0;
}

void saveImage() {
	float samples = iteration;
	// output image file
	image img(width, height);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int index = x + (y * width);
			glm::vec3 pix = renderState->image[index];
			img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
		}
	}

	std::string filename = renderState->imageName;
	std::ostringstream ss;
	ss << filename << "." << startTimeString << "." << samples << "samp";
	filename = ss.str();

	// CHECKITOUT
	img.savePNG(filename);
	//img.saveHDR(filename);  // Save a Radiance HDR file
}

void saveState() {
	std::cout << "Saving State" << std::endl;
	ofstream saveState;
	
	//saveState.open("../state/state.txt");
	saveState.open("../state/state.txt", std::ofstream::out | std::ofstream::trunc);
	int samples = iteration;
	std::cout << "Samples at begin save: " << samples << std::endl;
	saveState << "Samples: " << samples << "\n";
	for (int x = 0; x < renderState->image.size(); x++) {
		glm::vec3 pix = renderState->image[x];
		saveState << pix.x << " " << pix.y << " " << pix.z << "\n";
	}

	saveState.close();
	std::cout << "Samples at end save: " << iteration << std::endl;
}

void loadState() {
	string filename = "../state/state.txt";
	char* fname = (char*)filename.c_str();
	ifstream saveState;
	saveState.open(fname);
	if (!saveState.is_open()) {
		cout << "Error reading from file - aborting!" << endl;
		return;
	}

	renderState->usingSavedState = true;
	int lineNumber = 0;
	while (saveState.good()) {
		string line;
		utilityCore::safeGetline(saveState, line);
		if (!line.empty()) {
			vector<string> tokens = utilityCore::tokenizeString(line);
			if (lineNumber == 0) {
				iteration = atoi(tokens[1].c_str());
				//std::cout << atoi(tokens[0].c_str()) << std::endl;
			}
			else {
				float x = atof(tokens[0].c_str());
				float y = atof(tokens[1].c_str());
				float z = atof(tokens[2].c_str());
				glm::vec3 color = glm::vec3(x, y, z);
				renderState->image[lineNumber - 1] = color;
				//std::cout << lineNumber << " " << x << " " << y << " " << z << std::endl;
			}
		}
		lineNumber++;
	}
	saveState.close();
}

void runCuda() {
	if (camchanged) {
		iteration = 0;
		Camera& cam = renderState->camera;
		cameraPosition.x = zoom * sin(phi) * sin(theta);
		cameraPosition.y = zoom * cos(theta);
		cameraPosition.z = zoom * cos(phi) * sin(theta);

		cam.view = -glm::normalize(cameraPosition);
		glm::vec3 v = cam.view;
		glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
		glm::vec3 r = glm::cross(v, u);
		cam.up = glm::cross(r, v);
		cam.right = r;

		cam.position = cameraPosition;
		cameraPosition += cam.lookAt;
		cam.position = cameraPosition;
		camchanged = false;
	}

	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	if (iteration == 0) {
		pathtraceFree();
		pathtraceInit(scene);
#if LOADSTATE
		if (true) {
			loadState();
		}
#endif
	}

	if (iteration < renderState->iterations) {
		uchar4* pbo_dptr = NULL;
		iteration++;
		cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

		// execute the kernel
		int frame = 0;
		pathtrace(pbo_dptr, frame, iteration);

		// unmap buffer object
		cudaGLUnmapBufferObject(pbo);
	}
	else {
		saveImage();
		pathtraceFree();
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS) {
		switch (key) {
		case GLFW_KEY_ESCAPE:
			saveImage();
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_S:
			saveImage();
			break;
		case GLFW_KEY_X:
			saveState();
			break;
		case GLFW_KEY_SPACE:
			camchanged = true;
			renderState = &scene->state;
			Camera& cam = renderState->camera;
			cam.lookAt = ogLookAt;
			break;
		
		}
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (MouseOverImGuiWindow())
	{
		return;
	}
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
	middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
	if (leftMousePressed) {
		// compute new camera parameters
		phi -= (xpos - lastX) / width;
		theta -= (ypos - lastY) / height;
		theta = std::fmax(0.001f, std::fmin(theta, PI));
		camchanged = true;
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, zoom);
		camchanged = true;
	}
	else if (middleMousePressed) {
		renderState = &scene->state;
		Camera& cam = renderState->camera;
		glm::vec3 forward = cam.view;
		forward.y = 0.0f;
		forward = glm::normalize(forward);
		glm::vec3 right = cam.right;
		right.y = 0.0f;
		right = glm::normalize(right);

		cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
		cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
		camchanged = true;
	}
	lastX = xpos;
	lastY = ypos;
}
