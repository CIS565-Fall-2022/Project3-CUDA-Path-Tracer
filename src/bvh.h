#pragma once

#include "sceneStructs.h"

using namespace scene_structs;

class Bvh
{
private:

public:
  Bvh(const std::vector<Geom>& geoms);
  ~Bvh();
};

