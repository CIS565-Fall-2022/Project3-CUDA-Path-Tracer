#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "tiny_gltf.h"

static std::string GetFilePathExtension(const std::string& FileName) {
  if (FileName.find_last_of(".") != std::string::npos)
    return FileName.substr(FileName.find_last_of(".") + 1);
  return "";
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

    cout << "Materials: " << endl;
    for (int i = 0; i < materials.size(); i++) {
      cout << "ID: " << i << " use testure: " << materials[i].texID << endl;
    }

    cout << "Geometry: " << endl;
    for (int i = 0; i < geoms.size(); i++) {
      cout << "ID: " << i << " has texture? " << geoms[i].useTex << " has normal map? " << geoms[i].normalMapID << " has tangent? " << geoms[i].hasTangent << endl;
    }

    cout << "Texture Data: " << endl;
    for (int i = 0; i < textures.size(); i++) {
      cout << "ID: " << i << " normal offset:" << textures[i].offsetNormal << endl;
    }
    cout << "Total Texture Size: " << textures.size() << " Data: " << normalTexture.size() << endl;
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size() && false) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;
        string filename;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            } else if (strcmp(line.c_str(), "mesh") == 0) {
              cout << "Creating new mesh..." << endl;

              utilityCore::safeGetline(fp_in, line);
              if (!line.empty() && fp_in.good()) {
                vector<string> tokens = utilityCore::tokenizeString(line);
                if (strcmp(tokens[0].c_str(), "GLTF") == 0) {
                  cout << "Loading GLTF file" << endl;

                  //loadGLTF(tokens[1], newGeom);
                  filename = tokens[1];
                }
                else {
                  cout << "ERROR: Unknown file detected" << endl;
                }
              }
              else {
                cout << "ERROR: Fail to recognize" << endl;
              }
              newGeom.type = MESH;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());

            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        bool duplicate = false;

        if (geomIdMap.find(id) != geomIdMap.end()) {
          newGeom = geoms[geomIdMap[id]];
          duplicate = true;
        }
        else {
          geomIdMap[id] = geoms.size();
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
        if (duplicate) {
          newGeom.transform = newGeom.transform * newGeom.meshTransform;
        }

        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        if (newGeom.type == MESH && !duplicate) {
          loadGLTF(filename, newGeom);
        }
        else {
          geoms.push_back(newGeom);
        }

        return 1;
    }
}

int Scene::loadNodes(const vector<tinygltf::Node>& nodes, int index, glm::mat4& transform) {
  if (index >= nodes.size()) {
    cout << "ERROR: Node Index > Size" << endl;
    return 0;
  }

  cout << "Node ID: " << index << endl;
  
  glm::mat4 currTrans(1.f);

  auto currNode = nodes[index];
  if (!currNode.matrix.empty()) {
    currTrans = glm::make_mat4(currNode.matrix.data());
  }
  else {
    glm::vec3 t = nodes[index].translation.empty() ? glm::vec3(0.f) : glm::make_vec3(nodes[index].translation.data());
    glm::quat r = nodes[index].rotation.empty() ? glm::quat(1, 0, 0, 0) : glm::make_quat(nodes[index].rotation.data());
    glm::vec3 s = nodes[index].scale.empty() ? glm::vec3(1.f) : glm::make_vec3(nodes[index].scale.data());
    
    glm::mat4 translationMat = glm::translate(glm::mat4(), t);
    glm::mat4 rotationMat = glm::mat4_cast(r);
    glm::mat4 scaleMat = glm::scale(glm::mat4(), s);

    currTrans = translationMat * rotationMat * scaleMat;
  }

  currTrans = transform * currTrans;
  cout << "Node ID: " << index << endl;
  if (nodes[index].mesh != -1) {
    nodeToTransform[nodes[index].mesh] = currTrans;
  }

  for (int i = 0; i < nodes[index].children.size(); i++) {
    loadNodes(nodes, nodes[index].children[i], currTrans);
  }
  
  return 0;
}

int Scene::loadGLTF(const string& filename, Geom& geom) {
  cout << "Loading GLTF Mesh ..." << endl;

  tinygltf::Model model;
  tinygltf::TinyGLTF loader;
  std::string err;
  std::string warn;
  const std::string ext = GetFilePathExtension(filename);

  bool ret = false;
  if (ext.compare("glb") == 0) {
    // assume binary glTF.
    ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename.c_str());
  }
  else {
    // assume ascii glTF.
    ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename.c_str());
  }

  if (!warn.empty()) {
    std::cout << "glTF parse warning: " << warn << std::endl;
  }

  if (!err.empty()) {
    std::cerr << "glTF parse error: " << err << std::endl;
  }
  if (!ret) {
    std::cerr << "Failed to load glTF: " << filename << std::endl;
    return false;
  }

  
  loadNodes(model.nodes, model.scenes[0].nodes[0], glm::mat4());

  std::cout << "loaded glTF file has:\n"
    << model.accessors.size() << " accessors\n"
    << model.animations.size() << " animations\n"
    << model.buffers.size() << " buffers\n"
    << model.bufferViews.size() << " bufferViews\n"
    << model.materials.size() << " materials\n"
    << model.meshes.size() << " meshes\n"
    << model.nodes.size() << " nodes\n"
    << model.textures.size() << " textures\n"
    << model.images.size() << " images\n"
    << model.skins.size() << " skins\n"
    << model.samplers.size() << " samplers\n"
    << model.cameras.size() << " cameras\n"
    << model.scenes.size() << " scenes\n"
    << model.lights.size() << " lights\n";
  
  normalTextureID = colorTextureID = emissiveTextureID = -1;
  cout << "Texture Index: " << endl;
  for (int i = 0; i < model.materials.size(); i++) {
    auto material = model.materials[i];

    colorTextureID = material.pbrMetallicRoughness.baseColorTexture.index;
    if (colorTextureID != -1)
      colorTextureID += textures.size();

    normalTextureID = material.normalTexture.index;
    if (normalTextureID != -1)
      normalTextureID += textures.size();
    
    emissiveTextureID = material.emissiveTexture.index;
    if (emissiveTextureID != -1)
      emissiveTextureID += textures.size();
    
    emissiveFactor.x = material.emissiveFactor[0];
    emissiveFactor.y = material.emissiveFactor[1];
    emissiveFactor.z = material.emissiveFactor[2];

    cout << "Color: " << colorTextureID << " Normal: " << normalTextureID << endl;
    cout << "Emissive: " << emissiveTextureID << " Factors: " << emissiveFactor.x << " " << emissiveFactor.y << " " << emissiveFactor.z << endl << endl;
  }

  // Iterate through all texture declaration in glTF file
  for (const auto& gltfTexture : model.textures) {
    std::cout << "Found texture!";
    Texture loadedTexture;
    const auto& image = model.images[gltfTexture.source];
    loadedTexture.components = image.component;
    loadedTexture.width = image.width;
    loadedTexture.height = image.height;

    if (colorTextureID == textures.size()) {
      loadedTexture.offsetColor = colorTexture.size();
    }
    else if (normalTextureID == textures.size()) {
      loadedTexture.offsetNormal = normalTexture.size();
    }
    else if (emissiveTextureID == textures.size()) {
      loadedTexture.offsetEmissive = emissiveTexture.size();
    }

    const auto size =
      image.component * image.width * image.height * sizeof(unsigned char);

    cout << "Texture ID: " << textures.size() << endl;
    cout << "Width: " << image.width << " Height: " << image.height << " Components: " << image.component << endl;
    cout << "\nSize: " << size << "  Vector Size: " << image.image.size() / image.component << endl;
    cout << "Offset: " << loadedTexture.offsetNormal << endl;
    for (int i = 0; i < size; i += image.component) {
      //cout << image.image[i]/255.f << " ";
      glm::vec3 color;
      color.r = image.image[i + 0] / 255.f;
      color.g = image.image[i + 1] / 255.f;
      color.b = image.image[i + 2] / 255.f;
      // color.a = image.image[i + 3] / 255.f;

      if (colorTextureID == textures.size()) {
        colorTexture.push_back(color);
      }
      else if (normalTextureID == textures.size()) {
        normalTexture.push_back(color);
      }
      else if (emissiveTextureID == textures.size()) {
        emissiveTexture.push_back(color);
      }
    }
    cout << endl;
    //loadedTexture.image = new unsigned char[size];
    //memcpy(loadedTexture.image, image.image.data(), size);
    textures.push_back(loadedTexture);
  }

  std::cout << "Texture Number: " << textures.size() << endl;

  if (colorTextureID != -1) {
    Material newMaterial;
    // newMaterial.color = glm::vec3(.35, .85, .35);
    newMaterial.hasReflective = materials[geom.materialid].hasReflective;
    newMaterial.hasRefractive = materials[geom.materialid].hasRefractive;
    newMaterial.indexOfRefraction = materials[geom.materialid].indexOfRefraction;
    newMaterial.emittance = materials[geom.materialid].emittance;
    newMaterial.texID = colorTextureID;
    newMaterial.emissiveTexID = emissiveTextureID;

    newMaterial.color = materials[geom.materialid].color;

    geom.materialid = materials.size();

    materials.push_back(newMaterial);
  }

  // Iterate through all the meshes in the glTF file
  for(int meshId = 0; meshId < model.meshes.size(); meshId++){
    const auto& gltfMesh = model.meshes[meshId];
    std::cout << "\n================\nCurrent mesh has " << gltfMesh.primitives.size()
      << " primitives:\n";

    // Create a mesh object
    // Mesh<float> loadedMesh(sizeof(float) * 3);
    Geom newGeom;
    newGeom.startIndex = this->indices.size();
    newGeom.count = 0;
    newGeom.type = MESH;
    newGeom.materialid = geom.materialid;
    newGeom.hasNormalMap = normalTextureID;

    newGeom.meshTransform = nodeToTransform[meshId];
    newGeom.transform = geom.transform * nodeToTransform[meshId];
    newGeom.inverseTransform = glm::inverse(newGeom.transform);
    newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

    newGeom.boundingMax = glm::vec3(FLT_MIN);
    newGeom.boundingMin = glm::vec3(FLT_MAX);

    newGeom.normalMapID = normalTextureID;

    // To store the min and max of the buffer (as 3D vector of floats)
    glm::vec3 pMin, pMax;

    // Store the name of the glTF mesh (if defined)
    //loadedMesh.name = gltfMesh.name;
    
    // For each primitive
    for (const auto& meshPrimitive : gltfMesh.primitives) {
      // Boolean used to check if we have converted the vertex buffer format
      bool convertedToTriangleList = false;
      // This permit to get a type agnostic way of reading the index buffer

      const auto& indicesAccessor = model.accessors[meshPrimitive.indices];
      const auto& bufferView = model.bufferViews[indicesAccessor.bufferView];
      const auto& buffer = model.buffers[bufferView.buffer];
      const unsigned short* indices = reinterpret_cast<const unsigned short*>(&buffer.data[bufferView.byteOffset + indicesAccessor.byteOffset]);
      const auto byteStride = indicesAccessor.ByteStride(bufferView);
      const auto countIdx = indicesAccessor.count;

      std::cout << "Indices has count " << countIdx
        << " and stride " << byteStride << " bytes\n";
      for (int i = 0; i < countIdx; ++i) {
         //std::cout << indices[i] << " ";
        this->indices.push_back(indices[i]);
      }
      //std::cout << '\n';

      newGeom.count = this->indices.size() - newGeom.startIndex;

      /* ^^^^^ Done !!! ^^^^ */

      switch (meshPrimitive.mode) {
      case TINYGLTF_MODE_TRIANGLES:  // this is the simpliest case to handle

      {
        std::cout << "TRIANGLES\n";

        for (const auto& attribute : meshPrimitive.attributes) {
          const auto attribAccessor = model.accessors[attribute.second];
          const auto& bufferView = model.bufferViews[attribAccessor.bufferView];
          const auto& buffer = model.buffers[bufferView.buffer];
          const float* data = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + indicesAccessor.byteOffset]);
          const auto byte_stride = attribAccessor.ByteStride(bufferView);
          const auto count = attribAccessor.count;

          std::cout << "current attribute has count " << count
            << " and stride " << byte_stride << " bytes\n";

          std::cout << "attribute string is : " << attribute.first << '\n';
          if (attribute.first == "POSITION") {
            std::cout << "found position attribute\n";

            for (int i = 0; i < countIdx / 3; ++i) {
              // get the i'th triange's indexes
              auto f0 = indices[3 * i + 0];
              auto f1 = indices[3 * i + 1];
              auto f2 = indices[3 * i + 2];

              // get the 3 normal vectors for that face
              glm::vec3 v0, v1, v2;
              v0.x = data[f0 * 3 + 0];
              v0.y = data[f0 * 3 + 1];
              v0.z = data[f0 * 3 + 2];

              v1.x = data[f1 * 3 + 0];
              v1.y = data[f1 * 3 + 1];
              v1.z = data[f1 * 3 + 2];

              v2.x = data[f2 * 3 + 0];
              v2.y = data[f2 * 3 + 1];
              v2.z = data[f2 * 3 + 2];

              positions.push_back(v0);
              positions.push_back(v1);
              positions.push_back(v2);

              newGeom.boundingMin.x = min(min(v0.x, v1.x), min(v2.x, newGeom.boundingMin.x));
              newGeom.boundingMin.y = min(min(v0.y, v1.y), min(v2.y, newGeom.boundingMin.y));
              newGeom.boundingMin.z = min(min(v0.z, v1.z), min(v2.z, newGeom.boundingMin.z));
              newGeom.boundingMax.x = max(max(v0.x, v1.x), max(v2.x, newGeom.boundingMax.x));
              newGeom.boundingMax.y = max(max(v0.y, v1.y), max(v2.y, newGeom.boundingMax.y));
              newGeom.boundingMax.z = max(max(v0.z, v1.z), max(v2.z, newGeom.boundingMax.z));

            }
            cout << "Bounding Box: (" << newGeom.boundingMin.x << ", " << newGeom.boundingMax.x << ") (" 
              << newGeom.boundingMin.y << ", " << newGeom.boundingMax.y << ") (" 
              << newGeom.boundingMin.z << ", " << newGeom.boundingMax.z << ")" << endl;

            cout << "Positon Size: " << positions.size() << endl;
          }

          if (attribute.first == "NORMAL") {
            std::cout << "found normal attribute\n";

            // IMPORTANT: We need to reorder normals (and texture
            // coordinates into "facevarying" order) for each face

            // For each triangle :
            for (int i = 0; i < countIdx / 3; ++i) {
              // get the i'th triange's indexes
              auto f0 = indices[3 * i + 0];
              auto f1 = indices[3 * i + 1];
              auto f2 = indices[3 * i + 2];

              // get the 3 normal vectors for that face
              glm::vec3 n0, n1, n2;
              n0.x = data[f0 * 3 + 0];
              n0.y = data[f0 * 3 + 1];
              n0.z = data[f0 * 3 + 2];

              n1.x = data[f1 * 3 + 0];
              n1.y = data[f1 * 3 + 1];
              n1.z = data[f1 * 3 + 2];

              n2.x = data[f2 * 3 + 0];
              n2.y = data[f2 * 3 + 1];
              n2.z = data[f2 * 3 + 2];

              normals.push_back(n0);
              normals.push_back(n1);
              normals.push_back(n2);
            }
            cout << "Vector Size " << normals.size() << endl;
          }
          // Face varying comment on the normals is also true for the UVs
          if (attribute.first == "TEXCOORD_0") {
            std::cout << "Found texture coordinates\n";

            newGeom.useTex = 1;

            for (int i = 0; i < countIdx / 3; ++i) {
              // get the i'th triange's indexes
              auto f0 = indices[3 * i + 0];
              auto f1 = indices[3 * i + 1];
              auto f2 = indices[3 * i + 2];

              // get the texture coordinates for each triangle's
              // vertices
              glm::vec2 uv0, uv1, uv2;
              uv0.x = data[f0 * 2 + 0];
              uv0.y = data[f0 * 2 + 1];

              uv1.x = data[f1 * 2 + 0];
              uv1.y = data[f1 * 2 + 1];

              uv2.x = data[f2 * 2 + 0];
              uv2.y = data[f2 * 2 + 1];

              uvs.push_back(uv0);
              uvs.push_back(uv1);
              uvs.push_back(uv2);
              //cout << uv0.x << " " << uv0.y << " || ";
              //cout << uv1.x << " " << uv1.y << " || ";
              //cout << uv2.x << " " << uv2.y << endl;
            }

            for (int i = 0; i < uvs.size(); i++) {
              uvs[i].x = abs(uvs[i].x);
              uvs[i].y = abs(uvs[i].y);
            }
          }

          if (attribute.first == "TANGENT") {
            std::cout << "found tangent attribute\n";

            newGeom.hasTangent = 1;
            // IMPORTANT: We need to reorder normals (and texture
            // coordinates into "facevarying" order) for each face

            // For each triangle :
            for (int i = 0; i < countIdx / 3; ++i) {
              // get the i'th triange's indexes
              auto f0 = indices[3 * i + 0];
              auto f1 = indices[3 * i + 1];
              auto f2 = indices[3 * i + 2];

              // get the 3 normal vectors for that face
              glm::vec4 t0, t1, t2;
              t0.x = data[f0 * 4 + 0];
              t0.y = data[f0 * 4 + 1];
              t0.z = data[f0 * 4 + 2];
              t0.w = data[f0 * 4 + 3];

              t1.x = data[f1 * 4 + 0];
              t1.y = data[f1 * 4 + 1];
              t1.z = data[f1 * 4 + 2];
              t1.w = data[f1 * 4 + 3];

              t2.x = data[f2 * 4 + 0];
              t2.y = data[f2 * 4 + 1];
              t2.z = data[f2 * 4 + 2];
              t2.w = data[f2 * 4 + 3];

              tangents.push_back(t0);
              tangents.push_back(t1);
              tangents.push_back(t2);
            }

            //for (int i = 0; i < tangents.size(); i++) {
            //  cout << tangents[i].x << " " << tangents[i].y << " " << tangents[i].z << endl;
            //}

            cout << "Vector Size " << tangents.size() << endl;
          }
        }
        break;

      default:
        std::cerr << "primitive mode not implemented";
        break;
      }
      }

      //// TODO handle materials
      //for (size_t i{ 0 }; i < loadedMesh.faces.size(); ++i)
      //  loadedMesh.material_ids.push_back(materials->at(0).id);

      //meshes->push_back(loadedMesh);
      ret = true;
    }
    cout << "Load into Geometry list" << endl;
    geoms.push_back(newGeom);
  }

  return 1;
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
