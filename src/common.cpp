#include "common.h"

int Settings::traceDepth = 0;
int Settings::toneMapping = ToneMapping::ACES;
bool Settings::visualizeBVH = false;
bool Settings::sortMaterial = false;
bool Settings::singleKernel = false;

bool State::camChanged = true;