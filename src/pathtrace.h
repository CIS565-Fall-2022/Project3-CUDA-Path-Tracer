#pragma once

#include <device_launch_parameters.h>
#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathTrace(uchar4 *pbo, int frame, int iteration);
