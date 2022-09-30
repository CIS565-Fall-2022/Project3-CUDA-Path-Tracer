//#define _CRT_SECURE_NO_DEPRECATE
#include <ctime>
#include "main.h"
#include "preview.h"
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"
#include "rendersave.h"
#include "pathtrace.h"

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* guiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow;

std::string Preview::currentTimeString() {
	time_t now;
	time(&now);
	char buf[sizeof "0000-00-00_00-00-00z"];
	strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
	return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures() {
	glGenTextures(1, &displayImage);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void) {
	GLfloat vertices[] = {
		-1.0f, -1.0f,
		1.0f, -1.0f,
		1.0f,  1.0f,
		-1.0f,  1.0f,
	};

	GLfloat texcoords[] = {
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

	GLuint vertexBufferObjID[3];
	glGenBuffers(3, vertexBufferObjID);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(positionLocation);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(texcoordsLocation);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader() {
	const char* attribLocations[] = { "Position", "Texcoords" };
	GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
	GLint location;

	//glUseProgram(program);
	if ((location = glGetUniformLocation(program, "u_image")) != -1) {
		glUniform1i(location, 0);
	}

	return program;
}

void deletePBO(GLuint* pbo) {
	if (pbo) {
		// unregister this buffer object with CUDA
		cudaGLUnregisterBufferObject(*pbo);

		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glDeleteBuffers(1, pbo);

		*pbo = (GLuint)NULL;
	}
}

void deleteTexture(GLuint* tex) {
	glDeleteTextures(1, tex);
	*tex = (GLuint)NULL;
}

void cleanupCuda() {
	if (pbo) {
		deletePBO(&pbo);
	}
	if (displayImage) {
		deleteTexture(&displayImage);
	}
}

void initCuda() {
	cudaGLSetGLDevice(0);

	// Clean up on program exit
	atexit(cleanupCuda);
}

void initPBO() {
	// set up vertex data parameter
	int num_texels = width * height;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;

	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	glGenBuffers(1, &pbo);

	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject(pbo);

}

void errorCallback(int error, const char* description) {
	fprintf(stderr, "%s\n", description);
}

bool Preview::initImguiGL() {
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}

	window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetCursorPosCallback(window, mousePositionCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);

	// Set up GL context
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}
	printf("Opengl Version:%s\n", glGetString(GL_VERSION));
	//Set up ImGui

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	io = &ImGui::GetIO(); (void)io;
	ImGui::StyleColorsLight();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 120");

	return true;
}

GuiDataContainer* Preview::GetGUIData() {
	return guiData;
}

bool Preview::initBufs() {
	// Initialize other stuff
	initVAO();
	initTextures();
	initCuda();
	initPBO();
	GLuint passthroughProgram = initShader();

	glUseProgram(passthroughProgram);
	glActiveTexture(GL_TEXTURE0);

	return true;
}

void Preview::InitImguiData() {
	if (guiData) {
		delete guiData;
	}
	guiData = new GuiDataContainer();
}

// LOOK: Un-Comment to check ImGui Usage
void RenderImGui() {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	int lastScene;
	ImGui::Begin("Path Tracer Analytics", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
	{
		ImGui::Text("Traced Depth %d", guiData->traced_depth);
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

		lastScene = guiData->cur_scene;
		ImGui::Combo("Switch Scenes", &guiData->cur_scene, guiData->scene_file_names, guiData->num_scenes);

		if (PathTracer::isPaused()) {
			if (ImGui::Button("Resume Render")) {
				PathTracer::togglePause();
			}
		} else {
			if (ImGui::Button("Pause Render")) {
				PathTracer::togglePause();
			}
		}
		ImGui::Text(guiData->prompt_text);
		ImGui::InputText("Save File Name", guiData->buf, sizeof(guiData->buf));
		if (ImGui::Button("Save Render")) {
			if (strlen(guiData->buf) == 0) {
				guiData->prompt_text = "Please Enter a File Name";
			} else {
				static constexpr char const* illegals = "<>:\"/\\|?*";
				bool has_illegal = false;
				for (size_t i = 0; i < sizeof(illegals) && !has_illegal; ++i) {
					if (strchr(guiData->buf, illegals[i]))
						has_illegal = true;
				}

				if (has_illegal) {
					guiData->prompt_text = "Please Enter a Valid File Name";
				} else {
					if (!PathTracer::saveRenderState(guiData->buf)) {
						guiData->prompt_text = "Failed to Save! Some Error Occurred";
					}
				}
			}
		}
	}
	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	if (lastScene != guiData->cur_scene) {
		switchScene(guiData->scene_file_names[guiData->cur_scene]);
	}
}

/// <summary>
/// renders a pre-scene load menu
/// </summary>
void Preview::DoPreloadMenu() {
	bool load_scene = false, load_save = false;
	while (!load_scene && !load_save && !glfwWindowShouldClose(window)) {
		glfwPollEvents();

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("WELCOME", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		{
			ImGui::Combo("Scenes", &guiData->cur_scene, guiData->scene_file_names, guiData->num_scenes);
			if (ImGui::Button("Load Scene")) {
				load_scene = true;
			}

			ImGui::Combo("Saved Render Progress", &guiData->cur_save, guiData->save_file_names, guiData->num_saves);
			if (ImGui::Button("Load Saved Render")) {
				load_save = true;
			}
		}
		ImGui::End();

		ImGui::Render();

		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
	}

	if (!glfwWindowShouldClose(window)) {
		if (load_scene) {
			if (guiData->num_scenes > 0) {
				switchScene(guiData->scene_file_names[guiData->cur_scene]);
			} else {
				cout << "No Scene Found !!" << endl;
				exit(1);
			}
		} else {
			if (guiData->num_saves > 0) {
				int iter;
				if (read_state(guiData->save_file_names[guiData->cur_save], iter, scene)) {
					switchScene(scene, iter);
				}
			} else {
				cout << "No Saves Found !!" << endl;
				exit(1);
			}
		}
	} else {
		exit(0);
	}
}

bool Preview::CapturingMouse() {
	return io->WantCaptureMouse;
}
bool Preview::CapturingKeyboard() {
	return io->WantCaptureKeyboard;
}
void Preview::mainLoop() {
	bool once = true;
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		runCuda(once); once = false;

		string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
		glfwSetWindowTitle(window, title.c_str());
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glClear(GL_COLOR_BUFFER_BIT);

		// Binding GL_PIXEL_UNPACK_BUFFER back to default
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

		// Render ImGui Stuff
		RenderImGui();
		glfwSwapBuffers(window);
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();
}