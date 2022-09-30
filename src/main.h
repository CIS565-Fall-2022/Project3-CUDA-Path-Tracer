#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include "glslUtility.hpp"
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <vector>

#include "sceneStructs.h"
#include "image.h"
#include "pathtrace.h"
#include "utilities.h"
#include "scene.h"

using namespace std;

//-------------------------------
//----------PATH TRACER----------
//-------------------------------

constexpr char const* scene_files_dir = "../scenes/";
constexpr char const* save_files_dir = "../saves/";

extern Scene* scene;
extern int iteration;

extern int width;
extern int height;

bool switchScene(Scene* scene, int start_iter, bool from_save);
bool switchScene(char const* path);
void runCuda(bool init);
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);