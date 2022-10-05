#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <stb_image.h>
#include <stb_image_write.h>


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
            //loading OBJ files
            else if (strcmp(tokens[0].c_str(), "OBJECT_obj") == 0) {
                //loadObj(tokens[1].c_str());
                loadMesh(tokens[1].c_str());
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
        else if (strcmp(tokens[0].c_str(), "FOCAL") == 0) {
            camera.focalDistance = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "LENSE") == 0) {
            camera.lensRadius = atof(tokens[1].c_str());
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
            if (strcmp( tokens[0].c_str(), "RGB") == 0) {
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

            else if (strcmp(tokens[0].c_str(), "MICROFACET") == 0) {
                newMaterial.microfacet = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "ROUGHNESS") == 0) {
                newMaterial.roughness = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

//adapted from https://github.com/tinyobjloader/tinyobjloader/blob/master/loader_example.cc
//also https://vkguide.dev/docs/chapter-3/obj_loading/
int Scene::loadObj(const char* fileName)
{
    printf("loading OBJ file: %s\n", fileName);
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> m_materials;

    std::string warn;
    std::string err;

    char* material_dir = "../scenes";
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &m_materials, &warn, &err, fileName, material_dir,
        NULL, true);
    if (!warn.empty())
        std::cout << "WARN: " << warn << std::endl;
    if (!err.empty())
        std::cerr << "ERR: " << err << std::endl;
    if (!ret) {
        printf("Failed to load/parse .obj.\n");
        return false;
    }

    for (size_t i = 0; i < shapes.size(); i += 1) {
        printf("%s\n", shapes[i].name.c_str());
    }


    Geom geo;
    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
            geo.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
            geo.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
            geo.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "TEXTURE") == 0) {
            geo.textureName = tokens[1].c_str();
            loadTexture(geo, tokens[1].c_str());
        }

        utilityCore::safeGetline(fp_in, line);
    }

    geo.transform = utilityCore::buildTransformationMatrix(
        geo.translation, geo.rotation, geo.scale);
    geo.inverseTransform = glm::inverse(geo.transform);
    geo.invTranspose = glm::inverseTranspose(geo.transform);


    Obj obj;
    //For each shape
    for (size_t i = 0; i < shapes.size(); i++) {
        size_t index_offset = 0;
        // For each face
        for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
            //size_t fnum = shapes[i].mesh.num_face_vertices[f]; //should always be 3
            size_t fnum = 3; // hardcode loading to triangles

            geo.type = TRIANGLE;
            geo.materialid = materials.size();// + shapes[i].mesh.material_ids[f];
            // For each vertex in the face
            for (size_t v = 0; v < fnum; v++) {
                tinyobj::index_t idx = shapes[i].mesh.indices[index_offset + v];
                auto ver = static_cast<size_t>(idx.vertex_index);
                auto x = attrib.vertices[3 * ver + 0];
                auto y = attrib.vertices[3 * ver + 1];
                auto z = attrib.vertices[3 * ver + 2];
                geo.pos[v] = glm::vec3(x, y, z);


                if (idx.normal_index >= 0) {// -1 means no data
                    auto nor = static_cast<size_t>(idx.normal_index);
                    auto nx = attrib.normals[3 * nor + 0];
                    auto ny = attrib.normals[3 * nor + 1];
                    auto nz = attrib.normals[3 * nor + 2];
                    geo.normal[v] = glm::vec3(nx, ny, nz);
                }

                if (idx.texcoord_index >= 0) {
                    auto tex = static_cast<size_t>(idx.texcoord_index);
                    auto tx = attrib.texcoords[2 * tex + 0];
                    auto ty = attrib.texcoords[2 * tex + 1];
                    geo.uv[v] = glm::vec2(tx, ty);
                }

            }
            index_offset += fnum;

            geo.isObj = true;
            geoms.push_back(geo);

            Obj_geoms.push_back(geo);
            obj.box = Union(obj.box, AABB(geo.pos[0], geo.pos[1], geo.pos[2]));
        }
    }

    

    //load materials
    printf("material size: %d\n", m_materials.size());
    for (size_t i = 0; i < m_materials.size(); i++) {
        Material temp;

        //temp.name = m_materials[i].name.c_str();
        printf("material[%ld].name = %s\n", static_cast<long>(i),
            m_materials[i].name.c_str());

        temp.color = glm::vec3(
            static_cast<const double>(m_materials[i].diffuse[0]),
            static_cast<const double>(m_materials[i].diffuse[1]),
            static_cast<const double>(m_materials[i].diffuse[2]));
        //temp.color = glm::vec3(1, 1, 1);

        printf("  material.Kd = (%f, %f ,%f)\n",
            static_cast<const double>(m_materials[i].diffuse[0]),
            static_cast<const double>(m_materials[i].diffuse[1]),
            static_cast<const double>(m_materials[i].diffuse[2]));

        temp.specular.color = glm::vec3(
            static_cast<const double>(m_materials[i].specular[0]),
            static_cast<const double>(m_materials[i].specular[1]),
            static_cast<const double>(m_materials[i].specular[2]));
        temp.specular.exponent = 10; //hardcode exponent
        printf("  material.Ks = (%f, %f ,%f)\n",
            static_cast<const double>(m_materials[i].specular[0]),
            static_cast<const double>(m_materials[i].specular[1]),
            static_cast<const double>(m_materials[i].specular[2]));

        temp.hasReflective = 1;
        temp.hasRefractive = 1;
        temp.emittance = 0;
        temp.indexOfRefraction = 1.5;
        temp.microfacet = 1;
        temp.roughness = 0.5;

        //just hardcode, only work for simgle image
        temp.textureName = geo.textureName;
        temp.img = geo.img;
        temp.channels = geo.channels;
        temp.texture_width = geo.texture_width;
        temp.texture_height = geo.texture_height;
        
        materials.push_back(temp);
        OBJ_materials.push_back(temp);
    }

    return 1;

}


//refer to https://solarianprogrammer.com/2019/06/10/c-programming-reading-writing-images-stb_image-libraries/
glm::vec3 Scene::loadTexture(Geom& geo, const char* fileName)
{
    int width, height, channels;
    geo.img = stbi_load(fileName, &width, &height, &channels, 0);
    geo.texture_width = width;
    geo.texture_height = height;
    geo.channels = channels;
    if (geo.img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);

    
}


int Scene::loadMesh(const char* fileName)
{
    printf("loading OBJ file: %s\n", fileName);
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> m_materials;

    std::string warn;
    std::string err;

    char* material_dir = "../scenes";
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &m_materials, &warn, &err, fileName, material_dir,
        NULL, true);
    if (!warn.empty())
        std::cout << "WARN: " << warn << std::endl;
    if (!err.empty())
        std::cerr << "ERR: " << err << std::endl;
    if (!ret) {
        printf("Failed to load/parse .obj.\n");
        return false;
    }

    for (size_t i = 0; i < shapes.size(); i += 1) {
        printf("%s\n", shapes[i].name.c_str());
    }


    Geom geo;
    geo.type = MESH;
    geo.materialid = 6;

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
            geo.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
            geo.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
            geo.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "TEXTURE") == 0) {
            geo.textureName = tokens[1].c_str();
            loadTexture(geo, tokens[1].c_str());
        }

        utilityCore::safeGetline(fp_in, line);
    }

    geo.transform = utilityCore::buildTransformationMatrix(
        geo.translation, geo.rotation, geo.scale);
    geo.inverseTransform = glm::inverse(geo.transform);
    geo.invTranspose = glm::inverseTranspose(geo.transform);


    Geom triangle;
    triangle.transform = geo.transform;
    triangle.inverseTransform = geo.inverseTransform;
    triangle.invTranspose = geo.invTranspose;
    //For each shape
    for (size_t i = 0; i < shapes.size(); i++) {
        size_t index_offset = 0;
        // For each face
        for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
            //size_t fnum = shapes[i].mesh.num_face_vertices[f]; //should always be 3
            size_t fnum = 3; // hardcode loading to triangles

            triangle.type = TRIANGLE;
            triangle.materialid = materials.size();// + shapes[i].mesh.material_ids[f];
            // For each vertex in the face
            for (size_t v = 0; v < fnum; v++) {
                tinyobj::index_t idx = shapes[i].mesh.indices[index_offset + v];
                auto ver = static_cast<size_t>(idx.vertex_index);
                auto x = attrib.vertices[3 * ver + 0];
                auto y = attrib.vertices[3 * ver + 1];
                auto z = attrib.vertices[3 * ver + 2];
                triangle.pos[v] = glm::vec3(x, y, z);


                if (idx.normal_index >= 0) {// -1 means no data
                    auto nor = static_cast<size_t>(idx.normal_index);
                    auto nx = attrib.normals[3 * nor + 0];
                    auto ny = attrib.normals[3 * nor + 1];
                    auto nz = attrib.normals[3 * nor + 2];
                    triangle.normal[v] = glm::vec3(nx, ny, nz);
                }

                if (idx.texcoord_index >= 0) {
                    auto tex = static_cast<size_t>(idx.texcoord_index);
                    auto tx = attrib.texcoords[2 * tex + 0];
                    auto ty = attrib.texcoords[2 * tex + 1];
                    triangle.uv[v] = glm::vec2(tx, ty);
                }

            }
            index_offset += fnum;
            //push all triangles into Obj_geoms
            Obj_geoms.push_back(triangle);
            geo.bbox = Union(geo.bbox, AABB(triangle.pos[0], triangle.pos[1], triangle.pos[2]));
        }
    }

    //push mesh to geoms
    geoms.push_back(geo);


    //load materials
    printf("material size: %d\n", m_materials.size());
    for (size_t i = 0; i < m_materials.size(); i++) {
        Material temp;

        //temp.name = m_materials[i].name.c_str();
        printf("material[%ld].name = %s\n", static_cast<long>(i),
            m_materials[i].name.c_str());

        temp.color = glm::vec3(
            static_cast<const double>(m_materials[i].diffuse[0]),
            static_cast<const double>(m_materials[i].diffuse[1]),
            static_cast<const double>(m_materials[i].diffuse[2]));
        //temp.color = glm::vec3(1, 1, 1);

        printf("  material.Kd = (%f, %f ,%f)\n",
            static_cast<const double>(m_materials[i].diffuse[0]),
            static_cast<const double>(m_materials[i].diffuse[1]),
            static_cast<const double>(m_materials[i].diffuse[2]));

        temp.specular.color = glm::vec3(
            static_cast<const double>(m_materials[i].specular[0]),
            static_cast<const double>(m_materials[i].specular[1]),
            static_cast<const double>(m_materials[i].specular[2]));
        temp.specular.exponent = 10; //hardcode exponent
        printf("  material.Ks = (%f, %f ,%f)\n",
            static_cast<const double>(m_materials[i].specular[0]),
            static_cast<const double>(m_materials[i].specular[1]),
            static_cast<const double>(m_materials[i].specular[2]));

        temp.hasReflective = 1;
        temp.hasRefractive = 1;
        temp.emittance = 0;
        temp.indexOfRefraction = 1.5;
        temp.microfacet = 1;
        temp.roughness = 0.5;

        //just hardcode, only work for simgle image
        temp.textureName = triangle.textureName;
        temp.img = triangle.img;
        temp.channels = triangle.channels;
        temp.texture_width = triangle.texture_width;
        temp.texture_height = triangle.texture_height;

        materials.push_back(temp);
        OBJ_materials.push_back(temp);
    }

    return 1;

}
