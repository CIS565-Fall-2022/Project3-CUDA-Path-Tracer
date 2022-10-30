#include <iostream>
#include "scene.h"
#include <cstring>
#include <filesystem>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE

#include <stb_image.h>
#include <stb_image_write.h>

#include "tiny_gltf.h"

// source: https://stackoverflow.com/questions/20446201/how-to-check-if-string-ends-with-txt
bool has_suffix(const std::string& str, const std::string& suffix)
{
  return str.size() >= suffix.size() &&
    str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;

    if (!has_suffix(filename, "txt")) {
      loadTinyGltf(filename);
      return;
    }

    char* fname = (char*)filename.c_str();
    fp_in.open(fname);

    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        return 1;
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

// convert accessor index into float array
// error if accessor points to non-float data type
float* readBuffer(const tinygltf::Model& model, int accessorIdx, int* out_arraySize, const string &gltbDirectory) {
  using namespace tinygltf;

  const Accessor& accessor = model.accessors.at(accessorIdx);
  if (accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
    cout << "Trying to read non-float accessor meow" << endl;
  }

  int vectorSize = 0;

  switch (accessor.type) {
  case TINYGLTF_TYPE_VEC2:
    vectorSize = 2;
    break;
  case TINYGLTF_TYPE_VEC3:
    vectorSize = 3;
    break;
  default:
    cout << "Trying to read non-vec2/vec3 meow" << endl;
  }

  // need to allocate vectorSize * numOfVectors floats
  // eg. if we have 10 vec2s, we need 20 floats
  if (!(accessor.count > 0)) {
    cout << "Invalid accessor count meow" << endl;
  }
  int arraySize = vectorSize * accessor.count;
  float* result = new float[arraySize];

  const BufferView& bufferView = model.bufferViews.at(accessor.bufferView);
  const Buffer& buffer = model.buffers.at(bufferView.buffer);

  // open da buffer
  // byteStride option not supported for now
  ifstream fileStream;
  string relativePath = gltbDirectory + '\\' + buffer.uri;
  fileStream.open(relativePath, ios::binary); // try relative path first
  if (!fileStream) {
    // try direct path
    fileStream.open(buffer.uri);
    if (!fileStream) {
        cout << "Could not open file " << buffer.uri << " at relative (" << relativePath << ") or absolute paths" << endl;
    }
    else {
      cout << "Opened file at absolute path " << buffer.uri << endl;
    }
  }
  else {
    cout << "Opened file at relative path " << relativePath << endl;
  }

  if (!fileStream.seekg(bufferView.byteOffset, ios::beg)) {
    cout << "Error getting offset from file" << endl;
  }
  else {
    cout << bufferView.byteOffset << " bytes seeked successfully" << endl;
  }

  fileStream.read((char*) result, bufferView.byteLength);

  cout << "Just read " << fileStream.gcount() << " bytes from file (" << bufferView.byteLength
    << " were requested at offset " << bufferView.byteOffset << ")" << endl;

  fileStream.close();

  //for (int i = 0; i < arraySize; ++i) {
  //  printf(" %6.4lf", result[i]);
  //}

  return result;
}

// https://stackoverflow.com/questions/8518743/get-directory-from-file-path-c
string getDirectory(const string &filename) {
  size_t found = filename.find_last_of("/\\");
  return(filename.substr(0, found));
}

int Scene::loadTinyGltf(string filename) {
  using namespace tinygltf;

  Model model;
  TinyGLTF loader;
  std::string err;
  std::string warn;

  //bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
  bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);

  // for loading from relative paths
  string gltbDirectory = getDirectory(filename);

  if (!warn.empty()) {
    printf("Warn: %s\n", warn.c_str());
  }

  if (!err.empty()) {
    printf("Err: %s\n", err.c_str());
    return -1;
  }

  if (!ret) {
    printf("Failed to parse glTF\n");
    return -1;
  }

  cout << "Gltf read the json file successfully:" << filename << endl;

  cout << "MATERIALS PARSING-----" << endl;
  // materials will be ordered in scene array like original order
  // so indices from geometry will still line up
  for (int i = 0; i < model.materials.size(); ++i) {
    const tinygltf::Material& gltfMaterial = model.materials[i];
    scene_structs::Material newMaterial;
    // TODO: add support for textures, for now, it will just be a lambert
    newMaterial.color = glm::vec3(0.8f, 0.2f, 0.7f);
    newMaterial.emittance = 1.0f;

    cout << "Adding material #" << i << ": " << gltfMaterial.name << endl;
    materials.push_back(newMaterial);
  }

  cout << "GEOMETRY PARSING-----" << endl;
  for (const Node &node : model.nodes) {
    // for now, don't worry about transformations/scene graph structure on nodes
    int meshIdx = node.mesh;
    Mesh mesh = model.meshes.at(meshIdx);
    cout << "Gltf Parsing mesh " << mesh.name << endl;
    for (const Primitive &p : mesh.primitives) {
      // TODO: add other buffers here
      int posArrSize;
      int posAccessorIdx = p.attributes.at("POSITION");
      float* positionArray = readBuffer(model, posAccessorIdx, &posArrSize, gltbDirectory);
    }
  }
}
