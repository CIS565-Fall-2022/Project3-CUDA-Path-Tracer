//#define _CRT_SECURE_NO_DEPRECATE
#include "main.h"
#include "preview.h"
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"
#include "image.h"
#include "imageUtils.h"

#include "guiData.h"
#include "guiFileDialog.h"

#include "rendersave.h"
#include "pathtrace.h"
#include "Collision/DebugDrawer.h"
#include "Octree/octree.h"
#include "Denoise/denoise.h"

#include <thrust/execution_policy.h>

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* guiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow;

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

static void initTextures() {
	glGenTextures(1, &displayImage);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

static void initVAO(void) {
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

static GLuint initShader() {
	const char* attribLocations[] = { "Position", "Texcoords" };
	GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
	GLint location;

	//glUseProgram(program);
	if ((location = glGetUniformLocation(program, "u_image")) != -1) {
		glUniform1i(location, 0);
	}

	return program;
}

static void deletePBO(GLuint* pbo) {
	if (pbo) {
		// unregister this buffer object with CUDA
		cudaGLUnregisterBufferObject(*pbo);

		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glDeleteBuffers(1, pbo);

		*pbo = (GLuint)NULL;
	}
}

static void deleteTexture(GLuint* tex) {
	glDeleteTextures(1, tex);
	*tex = (GLuint)NULL;
}

static void cleanupCuda() {
	if (pbo) {
		deletePBO(&pbo);
	}
	if (displayImage) {
		deleteTexture(&displayImage);
	}
}

static void initCuda() {
	cudaGLSetGLDevice(0);

	// Clean up on program exit
	atexit(cleanupCuda);
}

static void initPBO() {
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
	ImGui::StyleColorsDark();
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
	if (!guiData) {
		guiData = new GuiDataContainer();
	} else {
		guiData->Reset();
	}

	if (guiData->denoiser_options.is_on) {
		PathTracer::enableDenoise();
	} else {
		PathTracer::disableDenoise();
	}
	PathTracer::setDenoise(guiData->desc);
}

static void RenderMainMenu() {
	std::string lastScene;
	ImGui::Text("Traced Depth %d", guiData->traced_depth);
	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

	lastScene = guiData->cur_scene;
	std::string filename;
	if (ImGui::OpenFileDialogue(guiData->scene_file_dialog, "Scene File: ", filename)) {
		guiData->cur_scene = filename;
		switchScene(filename.c_str());
	}
	if (ImGui::OpenFileDialogue(guiData->save_file_dialog, "Save File: ", filename)) {
		if (!PathTracer::saveRenderState(filename.c_str())) {
			std::cerr << "failed to save\n";
		}
	}
	if (ImGui::Button("Reload Scene")) {
		switchScene(guiData->cur_scene.c_str(), true);
	}
	if (PathTracer::isPaused()) {
		if (ImGui::Button("Resume Render")) {
			PathTracer::togglePause();
		}
	} else {
		if (ImGui::Button("Pause Render")) {
			PathTracer::togglePause();
		}
	}
}

static void RenderDenoiserMenu() {
	static constexpr char const* filter_options[Denoiser::FilterType::NUM_FILTERS] = {
		"A Trous",
		"Gaussian",
		"Simple Blur",
	};

	static constexpr char const* texture_options[DebugTextureType::NUM_OPTIONS] = {
		"None",
		"Show Normal Buffer",
		"Show Position Buffer",
		"Show Diffuse Buffer",
	};

	auto& ops = guiData->denoiser_options;
	auto& params = guiData->desc;

	ImGui::Checkbox("Denoise", &ops.is_on);
	ImGui::Checkbox("Use Diffuse Map", &params.use_diffuse);
	ImGui::Combo("Filter Type", (int*)&params.type, filter_options, Denoiser::FilterType::NUM_FILTERS);
	ImGui::SliderInt("Filter Size", &params.filter_size, 0, 100);
	ImGui::SliderFloat("Color Weight", &params.c_phi, 0.0f, 10.0f);
	ImGui::SliderFloat("Normal Weight", &params.n_phi, 0.0f, 10.0f);
	ImGui::SliderFloat("Position Weight", &params.p_phi, 0.0f, 10.0f);
	if (params.type == Denoiser::FilterType::GAUSSIAN) {
		ImGui::SliderFloat("Std. Dev.", &params.s_dev, 1.0f, 10.0f);
	}

	ImGui::Separator();

	std::string filename;
	if (ImGui::OpenFileDialogue(guiData->img_file_dialog, "Select a Reference Image", filename)) {
		guiData->ref_img = std::make_unique<Image>(filename);
		guiData->ref_img_file = std::move(filename);
	}

	guiData->NextBuf<bool>();
	ImGui::Checkbox("Display PSNR value", guiData->CurBuf<bool>().get());
	if (*guiData->CurBuf<bool>()) {
		if (guiData->ref_img) {
			DebugDrawer::DrawImage(guiData->ref_img_file.c_str(), 256, 256);
			Image img1{ width, height, PathTracer::getPBO() };
			float val = ImageUtils::CalculatePSNR(width * height, img1.getPixels(), guiData->ref_img->getPixels());
			ImGui::Text("PSNR = %f db", val);
		} else {
			ImGui::Text("No Reference Image");
		}
	}

	ImGui::Combo("Show Texture", &guiData->denoiser_options.debug_tex_idx, texture_options, DebugTextureType::NUM_OPTIONS);
	if (ops.is_on) {
		PathTracer::enableDenoise();
	} else {
		PathTracer::disableDenoise();
	}
	PathTracer::debugTexture((DebugTextureType)guiData->denoiser_options.debug_tex_idx);
	PathTracer::setDenoise(params);
}

static void RenderProfilingStats() {
	auto& data = PathTracer::GetProfileData();
	if (ImGui::BeginTable("profile data", 3)) {
		for (auto it = data.begin(); it != data.end(); ++it) {
			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			ImGui::Text("%s", it->first);

			ImGui::TableSetColumnIndex(1);
			ImGui::Text("%s", it->second.to_string().c_str());
		
			ImGui::TableSetColumnIndex(2);
			if (ImGui::Button("clear")) {
				it->second.clear();
			}
		}
		ImGui::EndTable();
	}
}

static void RenderDebugMenu() {
	if (ImGui::Button("Write PBO to Image")) {
		ImageUtils::SaveImage(PathTracer::getPBO());
	}

	bool draw_coord_frame, draw_debug_aabb, draw_world_aabb;

	guiData->NextBuf<bool>();
	if (!*guiData->CurBuf<bool>()) {
		if (ImGui::Button("[DEBUG] Draw World Frame")) {
			*guiData->CurBuf<bool>() = true;
		}
	} else {
		if (ImGui::Button("[DEBUG] Don't draw World Frame")) {
			*guiData->CurBuf<bool>() = false;
		}
	}
	draw_coord_frame = *guiData->CurBuf<bool>();

	guiData->NextBuf<bool>();
	if (!*guiData->CurBuf<bool>()) {
		if (ImGui::Button("[DEBUG] Draw AABB")) {
			*guiData->CurBuf<bool>() = true;
		}
	} else {
		if (ImGui::Button("[DEBUG] Don't draw AABB")) {
			*guiData->CurBuf<bool>() = false;
		}
	}
	draw_debug_aabb = *guiData->CurBuf<bool>();

	guiData->NextBuf<bool>();
	if (!*guiData->CurBuf<bool>()) {
		if (ImGui::Button("[DEBUG] Draw World AABB")) {
			*guiData->CurBuf<bool>() = true;
		}
	} else {
		if (ImGui::Button("[DEBUG] Don't draw World AABB")) {
			*guiData->CurBuf<bool>() = false;
		}
	}
	draw_world_aabb = *guiData->CurBuf<bool>();

	ImGui::Text("Octree Test");
	{
		ImGui::SliderInt("Octree Depth", &guiData->octree_depth, 0, 20);
		if (ImGui::Button("Generate Octree")) {
			if (guiData->test_tree) {
				delete guiData->test_tree;
				guiData->test_tree = nullptr;
			}

			guiData->test_tree = new octree(*g_scene, g_scene->world_AABB, guiData->octree_depth);
		}
		if (ImGui::Button("Pull Octree From GPU")) {
			if (guiData->test_tree) {
				delete guiData->test_tree;
				guiData->test_tree = nullptr;
			}

			guiData->test_tree = new octree(PathTracer::getTree());
		}
		if (guiData->test_tree) {
			if (ImGui::Button("Destroy Octree")) {
				delete guiData->test_tree;
				guiData->test_tree = nullptr;
			}
		}
	}
	if (draw_coord_frame) {
		float x = g_scene->world_AABB.min().x + 1.f;
		float y = g_scene->world_AABB.min().y + 1.f;
		glm::vec3 origin{ x, y, 0 };
		DebugDrawer::DrawLine3D(origin, origin + glm::vec3(2, 0, 0), { 1,0,0 });
		DebugDrawer::DrawLine3D(origin, origin + glm::vec3(0, 2, 0), { 0,1,0 });
		DebugDrawer::DrawLine3D(origin, origin + glm::vec3(0, 0, 2), { 0,0,1 });
	}
	if (draw_debug_aabb) {
		for (Geom const& g : g_scene->geoms) {
			if (g.type == MESH) {
				DebugDrawer::DrawAABB(g.bounds, { 1,0,0 });
			}
		}
	}
	if (draw_world_aabb) {
		DebugDrawer::DrawAABB(g_scene->world_AABB, { 0,0,1 });
	}

	if (guiData->test_tree) {
		if (ImGui::Button("Clear Depth Filter")) {
			if (guiData->octree_depth_filter != -1) {
				guiData->octree_depth_filter = -1;
			}
		}
		ImGui::SliderInt("Set Depth Filter (-1 means no)", &guiData->octree_depth_filter, -1, guiData->octree_depth);
		guiData->octree_intersection_cnt = 0;
		guiData->test_tree->dfs([&](node const& node, int depth) {
			if (guiData->octree_depth_filter != -1) {
				if (guiData->octree_depth_filter == depth) {
					DebugDrawer::DrawAABB(node.bounds, { 0,1,0 });
					guiData->octree_intersection_cnt += node.leaf_infos.size();
				}
			} else {
				DebugDrawer::DrawAABB(node.bounds, { 0,1,0 });
				guiData->octree_intersection_cnt += node.leaf_infos.size();
			}
			});
		std::string info = "intersection cnt: " + std::to_string(guiData->octree_intersection_cnt) +
			"\ntotal triangles: " + std::to_string(g_scene->triangles.size());

		ImGui::Text(info.c_str());
	}
}

static void RenderImGui() {
	// very ugly, I know
	extern Scene* g_scene;
	static constexpr float k_menu_width = 400.f;
	static constexpr float k_collapsable_width = 60.f;


	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	if (!ImGui::Begin("Path Tracer", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::End();
		return;
	}
	guiData->ResetBuf();
	ImGui::SetWindowSize(ImVec2(k_menu_width, 3 * k_collapsable_width));
	ImGui::Text("press H to hide GUI completely.");

	guiData->NextBuf<bool>();
	if (ImGui::IsKeyPressed('H')) {
		*guiData->CurBuf<bool>() = !*guiData->CurBuf<bool>();
	}
	if (*guiData->CurBuf<bool>()) {
		ImGui::SetNextWindowSize(ImVec2(-1, -1));
		return;
	}
	
	if (ImGui::CollapsingHeader("Main Menu")) {
		RenderMainMenu();
	}
	ImGui::Separator();
	if (ImGui::CollapsingHeader("Denoiser")) {
		RenderDenoiserMenu();
	}
	ImGui::Separator();
	if (ImGui::CollapsingHeader("Profiling Stats")) {
		RenderProfilingStats();
	}
	ImGui::Separator();
	if (ImGui::CollapsingHeader("Debug")) {
		RenderDebugMenu();
	}

	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

/// <summary>
/// renders a pre-scene load menu
/// </summary>
void Preview::DoPreloadMenu() {
	bool load_scene = false, load_save = false;
	std::string scene_file, save_file;
	ImGui::FileDialogue scene_dialogue("Select Scene File", false, ".txt");
	ImGui::FileDialogue save_dialogue("Select Save File", false, ".sav");

	while (!load_scene && !load_save && !glfwWindowShouldClose(window)) {
		glfwPollEvents();

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("WELCOME", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		{
			ImGui::Text("Choose a Scene File or a Save File to continue");
			ImGui::Separator();

			load_scene = ImGui::OpenFileDialogue(scene_dialogue, "Scene File Name: ", scene_file);
			load_save = ImGui::OpenFileDialogue(save_dialogue, "Save File Name: ", save_file);
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
			switchScene(scene_file.c_str());
			guiData->cur_scene = std::move(scene_file);

		} else {
			int iter;
			if (read_state(save_file.c_str(), iter, g_scene)) {
				switchScene(g_scene, iter, true, true);
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
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		runCuda();

		std::string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(g_iteration) + " Iterations";
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