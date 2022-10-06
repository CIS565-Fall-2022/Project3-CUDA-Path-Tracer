#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void firstTimePathTraceInit(Scene* scene);
void pathtraceFree();
void lastTimePathTraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
