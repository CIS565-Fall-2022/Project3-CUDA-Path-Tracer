#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>
#include <stb_image.h>

using namespace std;

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

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;
        string gltf_filename;

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
            else if (strcmp(line.c_str(), "gltf") == 0) {
                cout << "Loading new GLTF model..." << endl;
                newGeom.type = MESH;
            }
        }
        if (newGeom.type == MESH) {
            utilityCore::safeGetline(fp_in, gltf_filename);
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (newGeom.type != MESH || strcmp(tokens[1].c_str(), "-1") != 0) {
                newGeom.materialid = atoi(tokens[1].c_str());
                std::cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
            }
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

        if (newGeom.type == MESH) {
            loadGLTF(gltf_filename, newGeom);
        }

        geoms.push_back(newGeom);
        //cout << "mat size" << materials.size() << endl;
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
        for (int i = 0; i < 9; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            Texture* tex = nullptr;
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
                newMaterial.pbrMetallicRoughness.baseColorFactor = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());

                newMaterial.pbrMetallicRoughness.metallicFactor = newMaterial.hasReflective;
                newMaterial.pbrMetallicRoughness.roughnessFactor = 0.0;
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
                //newMaterial.emissiveFactor = newMaterial.emittance * newMaterial.color;
            }
            else if (strcmp(tokens[0].c_str(), "TEXTURE") == 0) {
                tex = &newMaterial.tex;
            }
            else if (strcmp(tokens[0].c_str(), "BUMP") == 0) {
                tex = &newMaterial.bump;
            }
            if (tex && strcmp(tokens[1].c_str(), "NONE") != 0) {
                int width, height, comps;
                unsigned char* pixels = stbi_load(tokens[1].c_str(), &width, &height, &comps, 3);
                
                if (!pixels) {
                    cout << "Image not loaded" << endl;
                }
                else {
                    tex->TexIndex = texData.size();
                    tex->width = width;
                    tex->height = height;
                    tex->size = width * height * 3;
                    tex->image = new unsigned char[tex->size];
                    for (int i = 0; i < width * height; ++i)
                    {
                        glm::vec3 col;
                        for (int j = 0; j < 3; ++j)
                        {
                            col[j] = (float)pixels[i * 3 + j] / 255.f;
                        }
                        texData.push_back(col);
                    }
                    //std::cout << texData.size() << endl;
                    //memcpy(tex->image, pixels, tex->size); some bugs in this so used texData
                    stbi_image_free(pixels);
                }
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

int Scene::loadGLTF(string filename, Geom& geom) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    bool ret;
    //Make sure we can read both glb and gltf
    string ext = filename.find_last_of(".") != string::npos ? filename.substr(filename.find_last_of(".") + 1) : "";
    if (ext.compare("glb") == 0) {
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
    }
    else {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    }
    if (!warn.empty()) {
        std::cout << "GLTF Warning!!" << warn << std::endl;
    }
    if (!err.empty()) {
        std::cout << "GLTF Error!!" << err << std::endl;
    }
    if (!ret) {
        std::cerr << "Failed to parse GLTF Model " << filename << endl;
        return -1;
    }

    /*std::cout << "loaded glTF file has:\n"
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
        << model.lights.size() << " lights\n";*/
    int texIndex = textures.size();
    int matIndex = materials.size();
    //cout << textures.size() << materials.size() << endl;
    //copy materials in model to host
    for (const tinygltf::Material& material : model.materials) {
        Material newMat;
        newMat.gltf = true;
        newMat.texIndex = texIndex;
        newMat.pbrMetallicRoughness.baseColorTexture = material.pbrMetallicRoughness.baseColorTexture;
        newMat.pbrMetallicRoughness.baseColorFactor = glm::make_vec3(material.pbrMetallicRoughness.baseColorFactor.data());
        newMat.pbrMetallicRoughness.metallicRoughnessTexture = material.pbrMetallicRoughness.metallicRoughnessTexture;
        newMat.pbrMetallicRoughness.metallicFactor = material.pbrMetallicRoughness.metallicFactor;
        newMat.pbrMetallicRoughness.roughnessFactor = material.pbrMetallicRoughness.roughnessFactor;
        newMat.normalTexture = material.normalTexture;
        newMat.emissiveFactor = glm::make_vec3(material.emissiveFactor.data());
        newMat.emissiveTexture = material.emissiveTexture;
        newMat.emittance = 0.f;
        //cout << newMat.emissiveTexture.index << endl;
        materials.push_back(newMat);
    }
    //copy textures in model to host
    for (const tinygltf::Texture& texture : model.textures) {
        Texture newTex;
        const tinygltf::Image& image = model.images[texture.source];
        newTex.components = image.component;
        newTex.width = image.width;
        newTex.height = image.width;
        newTex.size = image.component * image.width * image.height * sizeof(unsigned char);
        newTex.image = new unsigned char[newTex.size];
        memcpy(newTex.image, image.image.data(), newTex.size);
        textures.push_back(newTex);
    }
    //cout << textures.size() << materials.size() << endl;
    geom.primBegin = primitives.size();
    const unsigned short* indices = nullptr;
    const float* mesh_vertices = nullptr;
    const float* mesh_uvs = nullptr;
    const float* mesh_tangents = nullptr;
    const float* mesh_normal = nullptr;

    //Get all mesh
    for (const tinygltf::Mesh& mesh : model.meshes) {

        for (const tinygltf::Primitive& primitive : mesh.primitives) {
            //Create mesh object(learnt from Syoyo's https://github.com/syoyo/tinygltf/issues/71)
            const tinygltf::Accessor& accessor = model.accessors[primitive.indices];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            indices = reinterpret_cast<const unsigned short*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
            
            const tinygltf::Accessor& accessorPos = model.accessors[primitive.attributes.at("POSITION")];
            const tinygltf::BufferView& bufferViewPos = model.bufferViews[accessorPos.bufferView];
            const tinygltf::Buffer& bufferPos = model.buffers[bufferViewPos.buffer];
            mesh_vertices = reinterpret_cast<const float*>(&bufferPos.data[bufferViewPos.byteOffset + accessorPos.byteOffset]);
            if (primitive.attributes.count("NORMAL")) {
                const tinygltf::Accessor& accessorNormal = model.accessors[primitive.attributes.at("NORMAL")];
                const tinygltf::BufferView& bufferViewNormal = model.bufferViews[accessorNormal.bufferView];
                const tinygltf::Buffer& bufferNormal = model.buffers[bufferViewNormal.buffer];
                mesh_normal = reinterpret_cast<const float*>(&bufferNormal.data[bufferViewNormal.byteOffset + accessorNormal.byteOffset]);
            }
            if (primitive.attributes.count("TEXCOORD_0")) {
                const tinygltf::Accessor& accessorTex = model.accessors[primitive.attributes.at("TEXCOORD_0")];
                const tinygltf::BufferView& bufferViewTex = model.bufferViews[accessorTex.bufferView];
                const tinygltf::Buffer& bufferTex = model.buffers[bufferViewTex.buffer];
                mesh_uvs = reinterpret_cast<const float*>(&bufferTex.data[bufferViewTex.byteOffset + accessorTex.byteOffset]);
            }
            if (primitive.attributes.count("TANGENT")) {
                const tinygltf::Accessor& accessorTan = model.accessors[primitive.attributes.at("TANGENT")];
                const tinygltf::BufferView& bufferViewTan = model.bufferViews[accessorTan.bufferView];
                const tinygltf::Buffer& bufferTan = model.buffers[bufferViewTan.buffer];
                mesh_tangents = reinterpret_cast<const float*>(&bufferTan.data[bufferViewTan.byteOffset + accessorTan.byteOffset]);
            }
            //transfer all data to primitives
            for (size_t i = 0; i < accessor.count; i+=3) {
                Primitive prim;
                for (int j = 0; j < 3; j++) {
                    int idx = indices[i + j];
                    prim.pos[j] = glm::vec3(mesh_vertices[idx*3], mesh_vertices[idx*3 + 1], mesh_vertices[idx*3 + 2]);
                    if (mesh_normal) {
                        prim.hasNormal = true;
                        prim.normal[j] = glm::vec3(mesh_normal[idx*3], mesh_normal[idx*3 + 1], mesh_normal[idx*3 + 2]);
                    }
                    if (mesh_uvs) {
                        prim.hasUV = true;
                        prim.uv[j] = glm::vec2(mesh_uvs[idx*2], mesh_uvs[idx*2 + 1]);
                    }
                    if (mesh_tangents) {
                        prim.hasTangent = true;
                        prim.tangent[j] = glm::vec4(mesh_tangents[idx*4], mesh_tangents[idx*4 + 1], mesh_tangents[idx*4 + 2], mesh_tangents[idx*4 + 3]);
                    }
                    //need to do aabb here
                    glm::vec3 currentPos(geom.transform * glm::vec4(prim.pos[j], 1.f));
                    geom.aabb_min = glm::min(geom.aabb_min, currentPos);
                    geom.aabb_max = glm::max(geom.aabb_max, currentPos);
                }
                if (!mesh_normal)//init normal if original does not have one
                {
                    glm::vec3 normal = glm::normalize(glm::cross(prim.pos[1] - prim.pos[0], prim.pos[2] - prim.pos[0]));
                    for (int i = 0; i < 3; i++)
                    {
                        prim.normal[i] = normal;
                    }
                }
                prim.mat_id = matIndex + primitive.material;
                //cout << prim.mat_id;
                primitives.push_back(prim);
            }
        }
    }
    
    geom.primEnd = primitives.size();
    
    /*std::cout << geom.aabb_min.x << " " << geom.aabb_min.y << " " << geom.aabb_min.z << endl;
    std::cout << geom.aabb_max.x << " " << geom.aabb_max.y << " " << geom.aabb_max.z << endl;*/
    std::cout << primitives.size() << " " << geom.primBegin << " " << geom.primEnd << std::endl;
    return 1;
}
