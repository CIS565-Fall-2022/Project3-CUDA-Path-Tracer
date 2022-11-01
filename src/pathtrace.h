#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
void naiveIntegrator(int iter, int pixelcount, int traceDepth, int blockSize1d);
void directLightIntegrator(int iter, int pixelcount, int blockSize1d, int num_lights);