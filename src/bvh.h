#pragma once

#include "sceneStructs.h"

using namespace scene_structs;

class Bvh
{
private:

public:
  Bvh(const std::vector<Triangle>& triangles);
  ~Bvh();
};

