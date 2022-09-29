#pragma once

#include <device_launch_parameters.h>
#include <vector>
#include "scene.h"

class ToneMapping {
public:
    enum {
        None = 0, Filmic = 1, ACES = 2
    };
    static int method;
};

void InitDataContainer(GuiDataContainer* guiData);
void pathTraceInit(Scene *scene);
void pathTraceFree();
void pathTrace(uchar4 *pbo, int frame, int iteration);