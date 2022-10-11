#include "main.h"
#include "preview.h"
#include "Collision/DebugDrawer.h"
#include "consts.h"
#include <cstring>
#include <iostream>
#include <ctime>

static std::string startTimeString;

std::string scene_files_dir = "../scenes/";
std::string save_files_dir = "../saves/";

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;

JunksFromMain g_mainJunks;

Scene* g_scene;
RenderState* g_renderState;
int iteration;

int width;
int height;

std::string currentTimeString() {
	time_t now;
	time(&now);
	char buf[sizeof "0000-00-00_00-00-00z"];
	strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
	return std::string(buf);
}
//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {

#ifdef NO_DEFAULT_PATHS
	std::cout << "Enter Scene Files Dir: \n";
	std::cin >> scene_files_dir;
	std::cout << "Enter Save Files Dir: \n";
	std::cin >> save_files_dir;
#endif // NO_DEFAULT_PATHS

	startTimeString = currentTimeString();
	height = width = 800;

	// Initialize ImGui Data
	Preview::InitImguiData();
	// Initialize CUDA and GL components
	if (!Preview::initImguiGL()) {
		return EXIT_FAILURE;
	}
	// preload
	Preview::DoPreloadMenu();
	// GL bufs
	Preview::initBufs();

	PathTracer::unitTest();

	// GLFW main loop
	Preview::mainLoop();

	return EXIT_SUCCESS;
}
void resetCamState() {
	Camera& cam = g_renderState->camera;
	auto& cameraPosition = g_mainJunks.cameraPosition;
	auto& zoom = g_mainJunks.zoom;
	auto& phi = g_mainJunks.phi;
	auto& theta = g_mainJunks.theta;

	cameraPosition.x = zoom * sin(phi) * sin(theta);
	cameraPosition.y = zoom * cos(theta);
	cameraPosition.z = zoom * cos(phi) * sin(theta);

	cam.view = -glm::normalize(cameraPosition);
	glm::vec3 v = cam.view;
	glm::vec3 u = WORLD_UP;//glm::normalize(cam.up);
	glm::vec3 r = glm::cross(v, u);
	cam.up = glm::cross(r, v);
	cam.right = r;

	cam.position = cameraPosition;
	cameraPosition += cam.lookAt;
	cam.position = cameraPosition;
	camchanged = false;
}
bool switchScene(Scene* scene, int start_iter, bool from_save, bool force) {
	// Set up camera stuff from loaded path tracer settings
	auto& phi = g_mainJunks.phi;
	auto& theta = g_mainJunks.theta;
	auto& zoom = g_mainJunks.zoom;
	auto& cameraPosition = g_mainJunks.cameraPosition;
	auto& ogLookAt = g_mainJunks.ogLookAt;

	iteration = start_iter;
	g_renderState = &scene->state;

	if (!from_save) {
		Camera const& cam = g_renderState->camera;
		width = cam.resolution.x;
		height = cam.resolution.y;

		glm::vec3 view = cam.view;
		glm::vec3 up = cam.up;
		glm::vec3 right = glm::cross(view, up);
		up = glm::cross(right, view);


		// compute phi (horizontal) and theta (vertical) relative 3D axis
		// so, (0 0 1) is forward, (0 1 0) is up
		glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
		glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);

		cameraPosition = cam.position;
		phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
		theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
		ogLookAt = cam.lookAt;
		zoom = glm::length(cam.position - ogLookAt);
	}


	camchanged = true;
	leftMousePressed = rightMousePressed = middleMousePressed = false;
	lastX = lastY = 0;
	Preview::InitImguiData();

	resetCamState();
	PathTracer::pathtraceFree(scene, force);
	PathTracer::pathtraceInit(scene, g_renderState, force);

	return true;
}

bool switchScene(char const* path, bool force) {
	// Load scene file
	if (g_scene) {
		delete g_scene;
	}
	g_scene = new Scene(path);
	return switchScene(g_scene, 0, false, force);
}

void saveImage() {
	float samples = iteration;
	// output image file
	image img(width, height);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int index = x + (y * width);
			glm::vec3 pix = g_renderState->image[index];
			img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
		}
	}

	std::string filename = g_renderState->imageName;
	std::ostringstream ss;
	ss << filename << "." << startTimeString << "." << samples << "samp";
	filename = ss.str();

	// CHECKITOUT
	img.savePNG(filename);
	//img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda(bool init) {

	if (camchanged) {
		iteration = 0;

		if (!init) {
			// clear image buffer
			std::fill(g_renderState->image.begin(), g_renderState->image.end(), glm::vec3(0));
		}
		PathTracer::pathtraceFree(g_scene);
		PathTracer::pathtraceInit(g_scene, g_renderState);

		resetCamState();
	}

	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	if (iteration < g_renderState->iterations) {
		uchar4* pbo_dptr = NULL;
		cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

		// execute the kernel
		int frame = 0;
		iteration = PathTracer::pathtrace(pbo_dptr, iteration);

		// unmap buffer object
		cudaGLUnmapBufferObject(pbo);
	} else {
		saveImage();
		PathTracer::pathtraceFree(nullptr);
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	auto& ogLookAt = g_mainJunks.ogLookAt;

	if (action == GLFW_PRESS) {
		switch (key) {
		case GLFW_KEY_ESCAPE:
			saveImage();
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_S:
			if (!Preview::CapturingMouse() && !Preview::CapturingKeyboard()) {
				saveImage();
			}
			break;
		case GLFW_KEY_SPACE:
			camchanged = true;
			g_renderState = &g_scene->state;
			g_renderState->camera.lookAt = ogLookAt;
			break;
		case GLFW_KEY_P:
			PathTracer::togglePause();
			break;
		}
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (Preview::CapturingMouse()) {
		return;
	}
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
	middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (abs(xpos-lastX) < EPSILON || abs(ypos-lastY) < EPSILON)
		return; // otherwise, clicking back into window causes re-start

	auto& zoom = g_mainJunks.zoom;
	auto& phi = g_mainJunks.phi;
	auto& theta = g_mainJunks.theta;

	if (leftMousePressed) {
		// compute new camera parameters
		phi -= (xpos - lastX) / width;
		theta -= (ypos - lastY) / height;
		theta = std::fmax(0.001f, std::fmin(theta, PI));
		camchanged = true;
	} else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, zoom);
		camchanged = true;
	}  else if (middleMousePressed) {
		g_renderState = &g_scene->state;
		Camera& cam = g_renderState->camera;
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
