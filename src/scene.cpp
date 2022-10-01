#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <memory>

#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>
static std::string getFileExtension(const std::string Filename)
{
    if (Filename.find_last_of(".") != std::string::npos)
    {
        //offset
        return Filename.substr(Filename.find_last_of(".")+1);
    }
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
            } //Add gltf
            else if (strcmp(tokens[0].c_str(), "GLTF") == 0)
            {
                cout << "gltf triggered!" << endl;
                loadGLTF(tokens[1]);
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
//use tiny gltf
int Scene::loadGLTFNodes(const std::vector<tinygltf::Node>& nodes, const tinygltf::Node& node, bool* isLoaded,glm::mat4& previousMat)
{
    glm::mat4 tempMatrix;
    //traverse tree
    if (node.matrix.empty())
    {
        glm::vec3 translate = node.translation.empty() ? glm::vec3(0.f) : glm::vec3(node.translation[0], node.translation[1], node.translation[2]);
        glm::quat rotation = node.rotation.empty() ? glm::quat(1,0,0,0) : glm::quat(node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3]);
        glm::vec3 scale = node.scale.empty() ? glm::vec3(0.f) : glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
        tempMatrix = utilityCore::buildTransformationMatrix(translate, rotation, scale);
    }
    else
    {
        tempMatrix = glm::make_mat4(node.matrix.data());
    }
    tempMatrix = tempMatrix * previousMat;

    int meshOffset = meshes.size();

    for (int i=0;i<nodes.size();i++)
    {
        if (!isLoaded[i])
        {
            loadGLTFNodes(nodes, nodes[i], isLoaded, tempMatrix);
            isLoaded[i] = true;
        }
    }
    if (node.mesh == 0)
    {
        return 0;
    }
    //parse model data into personal structure
    Geom newGeom;
    newGeom.type = MESH;
    newGeom.mesh_id = meshes.size() + node.mesh;
    newGeom.materialid = 1;
    newGeom.transform = tempMatrix;
    newGeom.inverseTransform = glm::inverse(newGeom.transform);
    newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
    geoms.push_back(newGeom);
    return 1;

}
int Scene::loadGLTF(const std::string filename)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string error;
    std::string warn;
    cout << "Loading GLTF...." << endl;
    bool ret = false;
 
    string fileType = getFileExtension(filename);
    bool loadSuccess = false;
    if (fileType.compare("glb") == 0)
    {
        //same string
        //glb binary document
        ret=loader.LoadBinaryFromFile(&model, &error, &warn, filename);
    }
    else if (fileType.compare("gltf") == 0)
    {
        ret = loader.LoadASCIIFromFile(&model, &error, &warn, filename);
    }

    if (!warn.empty())
    {
        std::cout << "gltf parse warning:" << warn << std::endl;
    }

    if (!error.empty())
    {
        std::cout << "gltf parse error:" << error << std::endl;
    }
    if (!ret)
    {
        std::cerr << "failed to load gltf:" << filename << std::endl;
        return -1;
    }
    //Already got model
    //Check
    std::cout << "load gltf file has:\n"
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
        << model.cameras.size()<<" cameras\n"
        <<model.scenes.size()<<" Scenes\n"
        << model.lights.size() << " lights\n";

        //test pass
        //stored node as geometry
        //begin parsing
    bool* isNodeLoaded = new bool[model.nodes.size()];
    memset(isNodeLoaded, 0, sizeof(bool) * model.nodes.size());
    for (int i = 0; i < model.nodes.size(); i++)
    {
        if (!isNodeLoaded[i])
        {
            loadGLTFNodes(model.nodes,model.nodes[i],isNodeLoaded,glm::mat4(1.f));
            isNodeLoaded[i] = true;
        }
    }
    delete[] isNodeLoaded;

    int textureOffset = textures.size();
    int materialOffset = materials.size();
    //load material
    for (const tinygltf::Material& gltfMat : model.materials)
    {
        Material newMat;
        newMat.gltf = true;
        newMat.texOffset = textureOffset;
        newMat.pbrVal.baseColorTexture = gltfMat.pbrMetallicRoughness.baseColorTexture;
        newMat.pbrVal.metallicRoughnessTexture = gltfMat.pbrMetallicRoughness.metallicRoughnessTexture;
        newMat.pbrVal.metallicFactor = gltfMat.pbrMetallicRoughness.metallicFactor;
        newMat.pbrVal.roughnessFactor = gltfMat.pbrMetallicRoughness.roughnessFactor;
        newMat.normalTexture = gltfMat.normalTexture;
        newMat.emissiveFactor = glm::make_vec3(gltfMat.emissiveFactor.data());
        newMat.emissiveTexture = gltfMat.emissiveTexture;
        materials.push_back(newMat);
    }
    //Iterate every texture in gltf file
    for (const tinygltf::Texture& gltfTexture : model.textures)
    {
        Texture loadTexture;
        //const tinygltf::Image& image=model.images[gltfTexture.sampler]
        const tinygltf::Image& image = model.images[gltfTexture.source];
        loadTexture.component = image.component;
        loadTexture.height = image.height;
        loadTexture.width = image.width;
        loadTexture.size = image.component * image.height * image.width;
        loadTexture.image = new unsigned char[loadTexture.size];
        memcpy(loadTexture.image, image.image.data(), loadTexture.size);
        textures.push_back(loadTexture);
    }
    //get all meshes
    for (const tinygltf::Mesh gltfMesh : model.meshes)
    {
        std::cout << "Debug Message: current mesh has " << gltfMesh.primitives.size() <<" Primitives" << endl;
        Mesh loadMesh;
        loadMesh.prim_count = gltfMesh.primitives.size();
        loadMesh.prim_offset = primitives.size();
        //For each primitive
        for (const tinygltf::Primitive& meshPrimitives : gltfMesh.primitives)
        {
            Primitive prim;
            //create mesh object
            const tinygltf::Accessor& indicesAccessor = model.accessors[meshPrimitives.indices];
            const tinygltf::BufferView& bufferView = model.bufferViews[indicesAccessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            const uint16_t* indices = reinterpret_cast<const uint16_t*>(&buffer.data[bufferView.byteOffset + indicesAccessor.byteOffset]);

            const auto byteStride = indicesAccessor.ByteStride(bufferView);
            const size_t count = indicesAccessor.count;
            prim.count = count;
            prim.mat_id = meshPrimitives.material + materialOffset;

            //load indices
            prim.index_Offset = mesh_indices.size();
            for (int i = 0; i < indicesAccessor.count; i++)
            {
                mesh_indices.push_back(indices[i]);
            }
            std::cout << "\n";
            switch (meshPrimitives.mode)
            {
            case TINYGLTF_MODE_TRIANGLES:
            {
                std::cout << "Read Triangles" << endl;
                for (const auto& attribute : meshPrimitives.attributes)
                {
                    const tinygltf::Accessor& attributeAccessor = model.accessors[attribute.second];
                    const tinygltf::BufferView& bufferView = model.bufferViews[attributeAccessor.bufferView];
                    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

                }
            }
            }


        }

    }


}