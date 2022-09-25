#include <iostream>
#include "scene.h"
#include <cstring>
#include <map>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

std::map<std::string, int> MaterialTypeTokenMap = {
    { "Lambertian", Material::Type::Lambertian },
    { "MetallicWorkflow", Material::Type::MetallicWorkflow },
    { "Dielectric", Material::Type::Dielectric },
    { "Light", Material::Type::Light }
};

std::map<std::string, Model*> Resource::modelPool;
std::map<std::string, Image*> Resource::texturePool;

Model* Resource::loadModel(const std::string& filename) {
    auto find = modelPool.find(filename);
    if (find != modelPool.end()) {
        return find->second;
    }
    auto model = new Model;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::string warn, err;

    std::cout << "Model::loading {" << filename << "}" << std::endl;
    if (!tinyobj::LoadObj(&attrib, &shapes, nullptr, &warn, &err, filename.c_str())) {
        std::cout << "failed\n\tError msg {" << err << "}" << std::endl;
        return nullptr;
    }
    bool hasTexcoord = !attrib.texcoords.empty();

#if INDEXED_MESH_DATA
    model->vertices.resize(attrib.vertices.size() / 3);
    model->normals.resize(attrib.normals.size() / 3);
    memcpy(model->vertices.data(), attrib.vertices.data(), attrib.vertices.size() * sizeof(float));
    memcpy(model->normals.data(), attrib.normals.data(), attrib.normals.size() * sizeof(float));
    if (hasTexcoord) {
        model->texcoord.resize(attrib.texcoords.size() / 2);
        memcpy(model->texcoords.data(), attrib.texcoords.data(), attrib.texcoords.size() * sizeof(float));
    }
    else {
        model->texcoord.resize(attrib.vertices.size() / 3);
    }

    for (const auto& shape : shapes) {
        for (auto idx : shape.mesh.indices) {
            model->indices.push_back({ idx.vertex_index, idx.normal_index,
                hasTexcoord ? idx.texcoord_index : idx.vertex_index });
        }
    }
#else
    for (const auto& shape : shapes) {
        for (auto idx : shape.mesh.indices) {
            model->vertices.push_back(*((glm::vec3*)attrib.vertices.data() + idx.vertex_index));
            model->normals.push_back(*((glm::vec3*)attrib.normals.data() + idx.normal_index));

            model->texcoords.push_back(hasTexcoord ?
                *((glm::vec2*)attrib.texcoords.data() + idx.texcoord_index) :
                glm::vec2(0.f)
            );
        }
    }
#endif
    modelPool[filename] = model;
    return model;
}

Image* Resource::loadTexture(const std::string& filename) {
    auto find = texturePool.find(filename);
    if (find != texturePool.end()) {
        return find->second;
    }
    auto texture = new Image(filename);
    texturePool[filename] = texture;
    return texture;
}

void Resource::clear() {
    for (auto i : modelPool) {
        delete i.second;
    }
    modelPool.clear();

    for (auto i : texturePool) {
        delete i.second;
    }
    texturePool.clear();
}

Scene::Scene(const std::string& filename) {
    std::cout << "Scene::Reading from {" << filename << "}..." << std::endl;
    std::cout << " " << std::endl;
    char* fname = (char*)filename.c_str();
    fpIn.open(fname);
    if (!fpIn.is_open()) {
        std::cout << "Error reading from file - aborting!" << std::endl;
        throw;
    }
    while (fpIn.good()) {
        std::string line;
        utilityCore::safeGetline(fpIn, line);
        if (!line.empty()) {
            std::vector<std::string> tokens = utilityCore::tokenizeString(line);
            if (tokens[0] == "Material") {
                loadMaterial(tokens[1]);
                std::cout << " " << std::endl;
            } else if (tokens[0] == "Object") {
                loadModel(tokens[1]);
                std::cout << " " << std::endl;
            } else if (tokens[0] == "Camera") {
                loadCamera();
                std::cout << " " << std::endl;
            }
        }
    }
}

Scene::~Scene() {
}

void Scene::loadModel(const std::string& objId) {
    std::cout << "Scene::Loading Model {" << objId << "}..." << std::endl;

    ModelInstance instance;

    std::string line;
    utilityCore::safeGetline(fpIn, line);

    std::string filename = line;
    std::cout << "\tFrom file " << filename << std::endl;
    instance.meshData = Resource::loadModel(filename);

    //link material
    utilityCore::safeGetline(fpIn, line);
    if (!line.empty() && fpIn.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (materialMap.find(tokens[1]) == materialMap.end()) {
            std::cout << "\tMaterial {" << tokens[1] << "} doesn't exist" << std::endl;
            throw;
        }
        instance.materialId = materialMap[tokens[1]];
        std::cout << "\tLink to Material {" << tokens[1] << "(" << instance.materialId << ")}..." << std::endl;
    }

    //load transformations
    utilityCore::safeGetline(fpIn, line);
    while (!line.empty() && fpIn.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (tokens[0] == "Translate") {
            instance.translation = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }
        else if (tokens[0] == "Rotate") {
            instance.rotation = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }
        else if (tokens[0] == "Scale") {
            instance.scale = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }

        utilityCore::safeGetline(fpIn, line);
    }

    instance.transform = Math::buildTransformationMatrix(
        instance.translation, instance.rotation, instance.scale
    );
    instance.transfInv = glm::inverse(instance.transform);
    instance.normalMat = glm::transpose(glm::mat3(instance.transfInv));

    std::cout << "\tComplete" << std::endl;
    modelInstances.push_back(instance);
}

void Scene::loadCamera() {
    std::cout << "Scene::Loading Camera..." << std::endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 7; i++) {
        std::string line;
        utilityCore::safeGetline(fpIn, line);
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "Resolution") == 0) {
            camera.resolution.x = std::stoi(tokens[1]);
            camera.resolution.y = std::stoi(tokens[2]);
        } else if (tokens[0] == "FovY") {
            fovy = std::stof(tokens[1]);
        } else if (tokens[0] == "LensRadius") {
            camera.lensRadius = std::stof(tokens[1]);
        } else if (tokens[0] == "FocalDist") {
            camera.focalDist = std::stof(tokens[1]);
        } else if (tokens[0] == "Sample") {
            state.iterations = std::stoi(tokens[1]);
        } else if (tokens[0] == "Depth") {
            state.traceDepth = std::stoi(tokens[1]);
        } else if (tokens[0] == "File") {
            state.imageName = tokens[1];
        }
    }

    std::string line;
    utilityCore::safeGetline(fpIn, line);
    while (!line.empty() && fpIn.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (tokens[0] == "Eye") {
            camera.position = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        } else if (tokens[0] == "LookAt") {
            camera.lookAt = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        } else if (tokens[0] == "Up") {
            camera.up = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }
        utilityCore::safeGetline(fpIn, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (Pi / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / Pi;
    camera.fov = glm::vec2(fovx, fovy);
    camera.tanFovY = glm::tan(glm::radians(fovy * 0.5f));

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    std::cout << "\tComplete" << std::endl;
}

void Scene::loadMaterial(const std::string& matId) {
    std::cout << "Scene::Loading Material {" << matId << "}..." << std::endl;
    Material material;

    //load static properties
    for (int i = 0; i < 6; i++) {
        std::string line;
        utilityCore::safeGetline(fpIn, line);
        auto tokens = utilityCore::tokenizeString(line);
        if (tokens[0] == "Type") {
            material.type = MaterialTypeTokenMap[tokens[1]];
        }
        else if (tokens[0] == "BaseColor") {
            glm::vec3 baseColor(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
            material.baseColor = baseColor;
        }
        else if (tokens[0] == "Metallic") {
            material.metallic = std::stof(tokens[1]);
        }
        else if (tokens[0] == "Roughness") {
            material.roughness = std::stof(tokens[1]);
        }
        else if (tokens[0] == "Ior") {
            material.ior = std::stof(tokens[1]);
        }
        else if (tokens[0] == "Emittance") {
            material.emittance = std::stof(tokens[1]);
        }
    }
    materialMap[matId] = materials.size();
    materials.push_back(material);
    std::cout << "\tComplete" << std::endl;
}

//struct Material {
//    enum Type {
//        Lambertian = 0, MetallicWorkflow = 1, Dielectric = 2, Light = 3
//    };
//
//    glm::vec3 baseColor;
//    float metallic;
//    float roughness;
//    float ior;
//    float emittance;
//    int type;
//};