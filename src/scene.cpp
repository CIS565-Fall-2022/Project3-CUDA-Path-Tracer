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

Scene::Scene(std::string filename) {
    std::cout << "Reading scene from " << filename << " ..." << std::endl;
    std::cout << " " << std::endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        std::cout << "Error reading from file - aborting!" << std::endl;
        throw;
    }
    while (fp_in.good()) {
        std::string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            std::vector<std::string> tokens = utilityCore::tokenizeString(line);
            if (tokens[0] == "Material") {
                loadMaterial(tokens[1]);
                std::cout << " " << std::endl;
            } else if (tokens[0] == "Object") {
                loadGeom(tokens[1]);
                std::cout << " " << std::endl;
            } else if (tokens[0] == "Camera") {
                loadCamera();
                std::cout << " " << std::endl;
            }
        }
    }
}

int Scene::loadGeom(std::string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        std::cout << "ERROR: OBJECT ID does not match expected number of geoms" << std::endl;
        return -1;
    } else {
        std::cout << "Loading Geom " << id << "..." << std::endl;
        Geom newGeom;
        std::string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (line == "Sphere") {
                std::cout << "Creating new sphere..." << std::endl;
                newGeom.type = GeomType::Sphere;
            } else if (line == "Cube") {
                std::cout << "Creating new cube..." << std::endl;
                newGeom.type = GeomType::Cube;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            std::vector<std::string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialId = std::stoi(tokens[1]);
            std::cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialId << "..." << std::endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            std::vector<std::string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (tokens[0] == "Translate") {
                newGeom.translation = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
            } else if (tokens[0] == "Rotate") {
                newGeom.rotation = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
            } else if (tokens[0] == "Scale") {
                newGeom.scale = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = Math::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        return 1;
    }
}

int Scene::loadCamera() {
    std::cout << "Loading Camera ..." << std::endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        std::string line;
        utilityCore::safeGetline(fp_in, line);
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
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (tokens[0] == "Eye") {
            camera.position = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        } else if (tokens[0] == "LookAt") {
            camera.lookAt = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        } else if (tokens[0] == "Up") {
            camera.up = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }

        utilityCore::safeGetline(fp_in, line);
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

    std::cout << "Loaded camera!" << std::endl;
    return 1;
}

int Scene::loadMaterial(std::string matId) {
    int id = atoi(matId.c_str());
    if (id != materials.size()) {
        std::cout << "ERROR: MATERIAL ID does not match expected number of materials" << std::endl;
        return -1;
    } else {
        std::cout << "Loading Material " << id << "..." << std::endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 6; i++) {
            std::string line;
            utilityCore::safeGetline(fp_in, line);
            auto tokens = utilityCore::tokenizeString(line);
            if (tokens[0] == "Type") {
                newMaterial.type = MaterialTypeTokenMap[tokens[1]];
            }
            else if (tokens[0] == "BaseColor") {
                glm::vec3 baseColor(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
                newMaterial.baseColor = baseColor;
            } else if (tokens[0] == "Metallic") {
                newMaterial.metallic = std::stof(tokens[1]);
            } else if (tokens[0] == "Roughness") {
                newMaterial.roughness = std::stof(tokens[1]);
            } else if (tokens[0] == "Ior") {
                newMaterial.ior = std::stof(tokens[1]);
            } else if (tokens[0] == "Emittance") {
                newMaterial.emittance = std::stof(tokens[1]);
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
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
