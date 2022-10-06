#include "main.h"
#include "preview.h"
#include <cstring>
#include "glm/gtx/string_cast.hpp"
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
//#define TINYGLTF_NO_INCLUDE_STB_IMAGE
//#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#include "tiny_gltf.h"
static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static bool hasCheckpoint = false;
static bool saveCheckpoint = false;
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

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
	startTimeString = currentTimeString();

	if (argc < 2) {
		printf("Usage: %s SCENEFILE.txt\n", argv[0]);
		return 1;
	}

	const char* sceneFile = argv[1];
	char* checkpoint_folder = "";
	if (argc == 3)
	{
		checkpoint_folder = argv[2];
		hasCheckpoint = true;
		checkpoint_folder = "./checkpoint";
	}

	tinygltf::Model model;
	tinygltf::TinyGLTF loader;
	std::string err;
	std::string warn;

	bool gltf = loader.LoadASCIIFromFile(&model, &err, &warn, argv[1]);
	//bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, argv[1]); // for binary glTF(.glb)

	// Load scene file
	if (gltf)
	{
		scene = new Scene(model);
	}
	else {
		scene = new Scene(sceneFile);
	}

	//Create Instance for ImGUIData
	guiData = new GuiDataContainer();

	// Set up camera stuff from loaded path tracer settings
	iteration = 0;
	if (hasCheckpoint)
	{
		loadState(checkpoint_folder);
	}
	renderState = &scene->state;
	Camera& cam = renderState->camera;
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

void runCuda() {
	if (camchanged && !hasCheckpoint) {
		if (!hasCheckpoint)
		{
			iteration = 0;
		}
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
	}
	if (hasCheckpoint) {
		pathtraceFree();
		pathtraceInit(scene);
		pathtraceInitCheckpoint(scene);
		hasCheckpoint = false;
		camchanged = false;
	}
	if (saveCheckpoint)
	{
		saveState();
		saveCheckpoint = false;
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


void saveState()
{
	std::cout << "Begining Checkpoint" << std::endl;
	auto state = scene->state;
	auto camera = state.camera;
	std::ofstream image_output_file("./checkpoint/image.data");
	std::ostream_iterator<glm::vec3> output_iterator(image_output_file, ",");
	std::copy(state.image.begin(), state.image.end(), output_iterator);
	std::ofstream iterations_output_file("./checkpoint/iterations.data");
	iterations_output_file << iteration;
	std::ofstream camera_output_file("./checkpoint/camera.data");
	camera_output_file << camera.resolution << camera.position << camera.lookAt << camera.view << camera.up << camera.right << camera.fov << camera.pixelLength;
	//glm::ivec2 resolution;
	//glm::vec3 position;
	//glm::vec3 lookAt;
	//glm::vec3 view;
	//glm::vec3 up;
	//glm::vec3 right;
	//glm::vec2 fov;
	//glm::vec2 pixelLength;
	std::cout << "Checkpoint Complete" << std::endl;
	
}

void loadState(string checkpoint_folder)
{
	//GLM doesn't have an input stream operator
	std::cout << "Loading Checkpoint" << std::endl;
	std::ifstream image_file(checkpoint_folder + "/image.data");
	char first_bracket;
	double x_value;
	double y_value;
	double z_value;
	char second_bracket;
	char comma;
	std::vector<glm::vec3> image;
	while (image_file >> first_bracket)
	{
		
		image_file >> x_value >> comma >> y_value >> comma >> z_value >> second_bracket >> comma;
		glm::vec3 temp(x_value, y_value, z_value);
		image.push_back(temp);
	}
	scene->state.image = image;
	std::ifstream iterations_file(checkpoint_folder + "/iterations.data");
	iterations_file >> iteration;
	std::ifstream camera_file(checkpoint_folder + "/camera.data");
	glm::ivec2 resolution = utilityCore::readIVec2(camera_file);
	glm::vec3 position = utilityCore::readVec3(camera_file);
	glm::vec3 lookAt = utilityCore::readVec3(camera_file);
	glm::vec3 view = utilityCore::readVec3(camera_file);
	glm::vec3 up = utilityCore::readVec3(camera_file);
	glm::vec3 right = utilityCore::readVec3(camera_file);
	glm::vec2 fov = utilityCore::readVec2(camera_file);
	glm::vec2 pixelLength = utilityCore::readVec2(camera_file);

	scene->state.camera = { resolution, position, lookAt, view, up, right, fov, pixelLength};
	std::cout << "Loading Checkpoint Complete" << std::endl;
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
		case GLFW_KEY_C:
			saveCheckpoint = true;
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
