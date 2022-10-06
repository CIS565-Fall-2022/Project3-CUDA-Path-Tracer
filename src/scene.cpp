#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define LOAD_OBJ 1
#define USE_BB 1

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

int Scene::loadObj(const char* filepath, 
                   glm::mat4 transform,
                   glm::vec3 trans,
                   glm::vec3 rot,
                   glm::vec3 scale,
                   int mat_id,
                   std::vector<Triangle>* triangleArray,
                   const char* basepath,
                   bool triangulate) {
    tinyobj::attrib_t attrib;

    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filepath, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader error: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader warning: " << reader.Warning();
    }

    attrib = reader.GetAttrib();
    shapes = reader.GetShapes();

    glm::mat4 invTransform = glm::inverse(transform);
    glm::mat4 invTranspose = glm::inverseTranspose(transform);

    for (size_t s = 0; s < shapes.size(); s++) {

        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            // Loop over vertices in the face.
            for (size_t v = 1; v < fv - 1; v++) {
                // access to vertex

                // idxa is the primary vertex's idx
                tinyobj::index_t idxa = shapes[s].mesh.indices[index_offset];
                tinyobj::index_t idxb = shapes[s].mesh.indices[index_offset + v];
                tinyobj::index_t idxc = shapes[s].mesh.indices[index_offset + v + 1];

                tinyobj::real_t vxa = attrib.vertices[3 * size_t(idxa.vertex_index) + 0];
                tinyobj::real_t vya = attrib.vertices[3 * size_t(idxa.vertex_index) + 1];
                tinyobj::real_t vza = attrib.vertices[3 * size_t(idxa.vertex_index) + 2];

                tinyobj::real_t vxb = attrib.vertices[3 * size_t(idxb.vertex_index) + 0];
                tinyobj::real_t vyb = attrib.vertices[3 * size_t(idxb.vertex_index) + 1];
                tinyobj::real_t vzb = attrib.vertices[3 * size_t(idxb.vertex_index) + 2];

                tinyobj::real_t vxc = attrib.vertices[3 * size_t(idxc.vertex_index) + 0];
                tinyobj::real_t vyc = attrib.vertices[3 * size_t(idxc.vertex_index) + 1];
                tinyobj::real_t vzc = attrib.vertices[3 * size_t(idxc.vertex_index) + 2];


                tinyobj::real_t nxa = attrib.normals[3 * size_t(idxa.normal_index) + 0];
                tinyobj::real_t nya = attrib.normals[3 * size_t(idxa.normal_index) + 1];
                tinyobj::real_t nza = attrib.normals[3 * size_t(idxa.normal_index) + 2];

                tinyobj::real_t nxb = attrib.normals[3 * size_t(idxb.normal_index) + 0];
                tinyobj::real_t nyb = attrib.normals[3 * size_t(idxb.normal_index) + 1];
                tinyobj::real_t nzb = attrib.normals[3 * size_t(idxb.normal_index) + 2];

                tinyobj::real_t nxc = attrib.normals[3 * size_t(idxc.normal_index) + 0];
                tinyobj::real_t nyc = attrib.normals[3 * size_t(idxc.normal_index) + 1];
                tinyobj::real_t nzc = attrib.normals[3 * size_t(idxc.normal_index) + 2];

                // construct triangle object
                Vertex vertA = {
                    glm::vec4(vxa, vya, vza, 1),
                    glm::vec4(nxa, nya, nza, 0)
                };

                Vertex vertB = {
                    glm::vec4(vxb, vyb, vzb, 1),
                    glm::vec4(nxb, nyb, nzb, 0)
                };

                Vertex vertC = {
                    glm::vec4(vxc, vyc, vzc, 1),
                    glm::vec4(nxc, nyc, nzc, 0)
                };

                Triangle triangle = {
                    vertA,
                    vertB,
                    vertC
                };

                triangleArray->push_back(triangle);

            }
            index_offset += fv;
        }
    }

    return 1;
}

#if LOAD_OBJ
int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    }
    else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        bool containsObj = false;
        string objFilePath;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            cout << line.c_str() << endl;
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            }
            else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
            else if (strstr(line.c_str(), ".obj") != NULL) {
                cout << "Creating new OBJ..." << endl;
#if USE_BB
                newGeom.type = OBJ_BB;
#else
                newGeom.type = TRIANGLE;
#endif
                containsObj = true;
                objFilePath = line;
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
            }
            else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation, 
                                                                   newGeom.rotation, 
                                                                   newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
        newGeom.triangleNum = 0;

        if (containsObj) {
            std::vector<Triangle> triangleArray;
            loadObj(objFilePath.c_str(), 
                    newGeom.transform, 
                    newGeom.translation, 
                    newGeom.rotation, 
                    newGeom.scale, 
                    newGeom.materialid, 
                    &triangleArray,
                    NULL,
                    false);
#if USE_BB
            float min_x = FLT_MAX;
            float min_y = FLT_MAX;
            float min_z = FLT_MAX;
            float max_x = FLT_MIN;
            float max_y = FLT_MIN;
            float max_z = FLT_MIN;

            for (int i = 0; i < triangleArray.size(); i++) {
                Triangle tri = triangleArray[i];

                min_x = fmin(fmin(tri.a.pos[0], tri.b.pos[0]), fmin(tri.c.pos[0], min_x));
                max_x = fmax(fmax(tri.a.pos[0], tri.b.pos[0]), fmax(tri.c.pos[0], max_x));
                                                                      
                min_y = fmin(fmin(tri.a.pos[1], tri.b.pos[1]), fmin(tri.c.pos[1], min_y));
                max_y = fmax(fmax(tri.a.pos[1], tri.b.pos[1]), fmax(tri.c.pos[1], max_y));
                                                                        
                min_z = fmin(fmin(tri.a.pos[2], tri.b.pos[2]), fmin(tri.c.pos[2], min_z));
                max_z = fmax(fmax(tri.a.pos[2], tri.b.pos[2]), fmax(tri.c.pos[2], min_z));
            }

            BoundingBox box = {
                glm::vec3(min_x, min_y, min_z),
                glm::vec3(max_x, max_y, max_z)
            };

            newGeom.host_triangles = new Triangle[triangleArray.size()];
            newGeom.device_triangles = NULL;
            newGeom.triangleNum = triangleArray.size();

            newGeom.bb = box;
            for (int i = 0; i < triangleArray.size(); i++) {
                newGeom.host_triangles[i] = triangleArray[i];
            }

            geoms.push_back(newGeom);
#else
            for (int i = 0; i < triangleArray.size(); i++) {
                Triangle* triangleInObj = new Triangle(triangleArray[i]);

                Geom newTriangle = {
                    TRIANGLE,
                    newGeom.materialid,
                    newGeom.translation,
                    newGeom.rotation,
                    newGeom.scale,
                    newGeom.transform,
                    newGeom.inverseTransform,
                    newGeom.invTranspose,
                    triangleInObj,
                    NULL,
                    BoundingBox{
                    },
                    1,
                };
                geoms.push_back(newTriangle);
            }
#endif
        }
        else {
            geoms.push_back(newGeom);
        }
        return 1;
    }
}
#else
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
#endif

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
        for (int i = 0; i < 8; i++) {
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
            } else if (strcmp(tokens[0].c_str(), "ABSORPTION") == 0) {
                newMaterial.absorption = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

