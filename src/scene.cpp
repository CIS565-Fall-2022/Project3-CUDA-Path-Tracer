#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm\gtx\transform.hpp>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE

#define DEBUG_GLTF_TEXTURES 0

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

Scene::Scene(string filename, string gltfFilename):Scene(filename) {
  loadTinyGltf(gltfFilename);
  std::cout << "Building Bvh trees --------------------------" << std::endl;
#ifdef BVH
  bvh = Bvh(triangles, geoms);
#endif
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

// The gltf models I could find don't have a camera, so lets load some default values
void Scene::loadDefaultCamera() {
  RenderState& state = this->state;
  state.iterations = 5000;
  state.traceDepth = 8;
  state.imageName = "default_output_image";

  Camera& camera = state.camera;
  camera.resolution = glm::ivec2(800, 800);
  camera.position = glm::vec3(0.0, 5, 10.5);
  camera.lookAt = glm::vec3(0, 5, 0);
  camera.up = glm::vec3(0, 1, 0);

  float fovy = 45;
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
}

void updateTransformMats(Geom& geom) {
  geom.transform = glm::translate(geom.translation) * glm::mat4_cast(glm::quat(geom.rotation)) * glm::scale(geom.scale);
  geom.inverseTransform = glm::inverse(geom.transform);
  geom.invTranspose = glm::inverseTranspose(geom.transform);
}

void Scene::loadDefaultLight() {
  // Add these after gltf is loaded so the ids don't interfere
  Material lightMat;
  lightMat.color = glm::vec3(1);
  lightMat.emittance = 50;

  materials.push_back(lightMat);

  Geom square;
  square.type = CUBE;
  square.materialid = materials.size() - 1;
  square.translation = glm::vec3(0, 10, 0);
  square.rotation = glm::vec3(0);
  square.scale = glm::vec3(3, .3, 3);
  updateTransformMats(square);
  geoms.push_back(square);
}

// convert accessor index into float array OR int array
// Either way they will be arrays of 4 bytes, we can cast later
// error if accessor points to other data type
void* readBuffer(const tinygltf::Model& model, int accessorIdx, const string& gltbDirectory,
  int *out_arrayLength, int *out_componentType) {
  using namespace tinygltf;

  const Accessor& accessor = model.accessors.at(accessorIdx);
  int componentSizeInBytes = GetComponentSizeInBytes(accessor.componentType);
  int numComponents = GetNumComponentsInType(accessor.type);

  // need to allocate numComponents * numOfVectors floats
  // eg. if we have 10 vec2s, we need 20 floats
  if (!(accessor.count > 0)) {
    cout << "Invalid accessor count meow" << endl;
  }
  int arraySizeInBytes = componentSizeInBytes * numComponents * accessor.count;
  char* result = new char[arraySizeInBytes]; // we can cast this to int, float, etc. for index buffers... very sketchy

  const BufferView& bufferView = model.bufferViews.at(accessor.bufferView);
  const Buffer& buffer = model.buffers.at(bufferView.buffer);

  // open da buffer
  // byteStride option not supported for now
  ifstream fileStream;
  string relativePath = gltbDirectory + '\\' + buffer.uri;
  fileStream.open(relativePath, ios::binary); // try relative path first
  if (!fileStream) {
    // try direct path
    fileStream.open(buffer.uri, ios::binary);
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

  fileStream.read((char*)result, bufferView.byteLength);

  cout << "Just read " << fileStream.gcount() << " bytes from file (" << bufferView.byteLength
    << " were requested at offset " << bufferView.byteOffset << ")" << endl;

  fileStream.close();

  //for (int i = 0; i < arraySizeInBytes; ++i) {
  //  printf(" %6.4lf", result[i]);
  //}
  
  // number of elements in array, not accounting for grouping by components
  *out_arrayLength = numComponents * accessor.count;
  if (out_componentType != NULL) {
    *out_componentType = accessor.componentType;
  }
  return (void *) result;
}

void appendVec4Buffer(vector<glm::vec4>& vec4Buffer, const float* elts, int eltsLength) {
  for (int i = 0; i + 3 < eltsLength; i = i + 4) {
    glm::vec4 elt(elts[i], elts[i + 1], elts[i + 2], elts[i + 3]);
    vec4Buffer.push_back(elt);
  }
}

void appendVec3Buffer(vector<glm::vec3> &vec3Buffer, const float* elts, int eltsLength) {
  for (int i = 0; i + 2 < eltsLength; i = i + 3) {
    glm::vec3 elt(elts[i], elts[i + 1], elts[i + 2]);
    vec3Buffer.push_back(elt);
  }
}

void appendVec2Buffer(vector<glm::vec2>& vec2Buffer, const float* elts, int eltsLength) {
  for (int i = 0; i + 1 < eltsLength; i = i + 2) {
    glm::vec2 elt(elts[i], elts[i + 1]);
    vec2Buffer.push_back(elt);
  }
}

// need to deal with a bit differently depending on if int or short
// official spec says indices should be an int array
// but it's a short array in the avocado file
void appendIndicesBuffer(vector<unsigned int>& indicesBuffer, void *elts, int eltsLength, int gltf_component_type) {
  if (gltf_component_type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
    unsigned int* indices = (unsigned int*)elts;
    for (int i = 0; i < eltsLength; ++i) {
      indicesBuffer.push_back(indices[i]);
    }
  }
  else if (gltf_component_type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
    unsigned short* indices = (unsigned short*)elts;
    for (int i = 0; i < eltsLength; ++i) {
      indicesBuffer.push_back(indices[i]);
    }
  }
  else {
    cout << "Unsupported component type for indices, gtfo" << endl;
  }
}

// https://stackoverflow.com/questions/8518743/get-directory-from-file-path-c
string getDirectory(const string &filename) {
  size_t found = filename.find_last_of("/\\");
  return(filename.substr(0, found));
}

glm::mat4 getTransforms(glm::mat4 parentTransform, const tinygltf::Node& node,
  glm::vec3 &out_translation, glm::vec3 &out_rotation, glm::vec3 &out_scale) {
  
  // According to gltf spec, matrix overrides translate/rotate/scale properties
  if (node.matrix.size() == 16) { 
    glm::mat4 localTransform = glm::make_mat4(&node.matrix[0]);
    // The global transformation matrix of a node is the product of the global transformation matrix
    // of its parent node and its own local transformation matrix
    glm::mat4 transform = parentTransform * localTransform;

    // get components from matrix
    glm::quat rotationQuat;
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(transform, out_scale, rotationQuat, out_translation, skew, perspective);
    rotationQuat = glm::conjugate(rotationQuat);
    out_rotation = glm::eulerAngles(rotationQuat);

    return transform;
  }

  // No matrix case, use translate/rotate/scale
  if (node.translation.size() >= 3) {
    out_translation = glm::make_vec3(&node.translation[0]);
  }
  else {
    out_translation = glm::vec3(0);
  }

  if (node.scale.size() >= 3) {
    out_scale = glm::make_vec3(&node.scale[0]);
  }
  else {
    out_scale = glm::vec3(1);
  }

  glm::quat rotationQuat;
  if (node.rotation.size() >= 4) {
    rotationQuat = glm::make_quat(&node.rotation[0]);
  }
  else {
    rotationQuat = glm::quat(glm::vec3(0));
  }
  out_rotation = glm::eulerAngles(rotationQuat);

  // use as transform matrix T * R * S
  glm::mat4 localTransform = glm::translate(out_translation) * glm::mat4_cast(rotationQuat) * glm::scale(out_scale);
  return parentTransform * localTransform;
}

void loadNode(int nodeIdx, const tinygltf::Model &model, string gltbDirectory,
  const glm::mat4 &parentTransform, std::vector<Geom> &geoms, std::vector<Triangle> &triangles, int materialOffset) {
  using namespace tinygltf;

  const Node& node = model.nodes.at(nodeIdx);

  glm::vec3 translation, rotation, scale;
  glm::mat4 transform, invTransform, invTranspose;

  transform = getTransforms(parentTransform, node, translation, rotation, scale);
  invTransform = glm::inverse(transform);
  invTranspose = glm::inverseTranspose(transform);

  for (const int& childNodeIdx : node.children) {
    loadNode(childNodeIdx, model, gltbDirectory, transform, geoms, triangles, materialOffset);
  }

  if (node.mesh == -1) {
    // If it's a leaf node, we have no way to render it, so error
    if (node.children.size() == 0) {
      cout << "error Cannot render leaf node" << endl;
    }
    return;
  }

  Mesh mesh = model.meshes.at(node.mesh);
  cout << "Gltf Parsing mesh " << mesh.name << endl;

  InputMesh newMesh;

  for (const Primitive& p : mesh.primitives) {
    // TODO: add other buffers here
    int posArrLength, normArrLength, uvArrLength, tangentArrLength, indicesArrLength, indicesComponentType;

    float* positionArray = (float*)readBuffer(model, p.attributes.at("POSITION"), gltbDirectory, &posArrLength, NULL);
    float* normalArray = (float*)readBuffer(model, p.attributes.at("NORMAL"), gltbDirectory, &normArrLength, NULL);
    if (posArrLength != normArrLength) {
      cout << "Warning - Positions buffer length not equal to normals buffer length" << endl;
    }
    appendVec3Buffer(newMesh.positions, positionArray, posArrLength);
    appendVec3Buffer(newMesh.normals, normalArray, normArrLength);

    if (p.attributes.count("TANGENT")) {
      float* tangentArray = (float*)readBuffer(model, p.attributes.at("TANGENT"), gltbDirectory, &tangentArrLength, NULL);
      appendVec4Buffer(newMesh.tangents, tangentArray, tangentArrLength);
      free(tangentArray);
    }

    if (p.attributes.count("TEXCOORD_0")) {
      float* uvArray = (float*)readBuffer(model, p.attributes.at("TEXCOORD_0"), gltbDirectory, &uvArrLength, NULL);
      appendVec2Buffer(newMesh.uvCoords, uvArray, uvArrLength);
      free(uvArray);
    }

    void* indicesArray = (void*)readBuffer(model, p.indices, gltbDirectory, &indicesArrLength, &indicesComponentType);
    appendIndicesBuffer(newMesh.indices, indicesArray, indicesArrLength, indicesComponentType);

    free(positionArray);
    free(normalArray);
    free(indicesArray);

    glm::vec3 default_translation = glm::vec3(0);
    glm::vec3 default_rotation = glm::vec3(0);
    glm::vec3 default_scale = glm::vec3(10);

    // Each primitive is 1 mesh
    Geom mesh;
    mesh.type = TRIANGLE_MESH;
    mesh.translation = translation;
    mesh.rotation = rotation;
    mesh.scale = scale;
    mesh.materialid = materialOffset + p.material;
    mesh.transform = transform;
    mesh.inverseTransform = invTransform;
    mesh.invTranspose = invTranspose;
    
    // Now parse the triangles of the mesh
    mesh.triangleOffset = triangles.size();

    int indicesLen = newMesh.indices.size();
    for (int i = 0; i + 2 < indicesLen; i = i + 3) {
      Triangle triangle;

      for (int idx = i; idx < i + 3; ++idx) {
        Vertex v;
        v.position = newMesh.positions.at(newMesh.indices.at(idx));
        v.normal = newMesh.normals.at(newMesh.indices.at(idx));
        if (p.attributes.count("TEXCOORD_0")) {
          v.uv = newMesh.uvCoords.at(newMesh.indices.at(idx));
        }
        if (p.attributes.count("TANGENT")) {
          v.tangent = newMesh.tangents.at(newMesh.indices.at(idx));
        }
        triangle.verts[idx - i] = v;
      }

      triangles.push_back(triangle);
    }

    mesh.numTriangles = triangles.size() - mesh.triangleOffset;
    geoms.push_back(mesh);
  }
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

  cout << "TEXTURES PARSING------" << endl;
  for (const tinygltf::Image& imageSource : model.images) {

    // For simplicity, assume all textures are complete images, with no special sampler
    // Thus, textures and images have a 1 to 1 correspondence
    scene_structs::Image newImage;
    newImage.height = imageSource.height;
    newImage.width = imageSource.width;

    // images will be vector of color channel data
    // Read the rgb channels
    if (imageSource.component < 3) { // component is the number of channels per pixel
      cout << "UNSUPPORTED - image source does not have complete rgba data" << endl;
    }
    for (int i = 0; i < imageSource.image.size(); i += imageSource.component) {
      const float r = (int) imageSource.image.at(i) / 255.0f; // convert char rgb to float rgb
      const float g = (int) imageSource.image.at(i + 1) / 255.0f;

      const float b = (int) imageSource.image.at(i + 2) / 255.0f;
      newImage.pixels.push_back(glm::vec3(r, g, b));
    }

    cout << "Adding image: " << imageSource.uri << endl;
    images.push_back(newImage);
  }

  cout << "MATERIALS PARSING-----" << endl;
  int materialOffset = materials.size(); // offset indices of previously loaded materials, eg. from cornell box

  // materials will be ordered in scene array like original order
  // so indices from geometry will still line up
  for (int i = 0; i < model.materials.size(); ++i) {
    const tinygltf::Material& gltfMaterial = model.materials[i];
    scene_structs::Material newMaterial;

    int textureIndex = gltfMaterial.pbrMetallicRoughness.baseColorTexture.index;
    if (textureIndex == -1 || DEBUG_GLTF_TEXTURES) {
      cout << "Material has no base color texture. Will render using base color factor (default is white)" << endl;
      auto& color = gltfMaterial.pbrMetallicRoughness.baseColorFactor;
      newMaterial.color = glm::vec3(color[0], color[1], color[2]);
    }
    else {
      cout << "Material has image base color texture" << endl;
      newMaterial.colorImageId = model.textures.at(textureIndex).source;
    }

    if (!DEBUG_GLTF_TEXTURES) {
      newMaterial.normalMapImageId = gltfMaterial.normalTexture.index;
    }
    newMaterial.hasReflective = 0;

    newMaterial.emittance = 0; // all gltf scenes tested don't have emittance

    cout << "Adding material #" << i << ": " << gltfMaterial.name << endl;
    materials.push_back(newMaterial);
  }

  cout << "GEOMETRY PARSING-----" << endl;
  const tinygltf::Scene& scene = model.scenes.at(model.defaultScene);

  for (const int &nodeIdx : scene.nodes) {
    loadNode(nodeIdx, model, gltbDirectory, glm::mat4(1.0), this->geoms, this->triangles, materialOffset);
  }
  return 0;
}
