#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "stb_image.h"
#include "stb_image_write.h"

#include "tiny_obj_loader.h"

#define LOAD_OBJ 1

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
            }
            else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "TEXTURE") == 0) {
                loadTexture(tokens[1]);
                cout << " " << endl;
            }
        }
    }
}

#if LOAD_OBJ
// example code taken from https://github.com/tinyobjloader/tinyobjloader
int Scene::loadObj(const char* filename, 
    std::vector<Triangle>* triangleArray,
    const char* basepath = NULL,
    bool triangulate = true)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "../Project3-CUDA-Path-Tracer/objs/"; // Path to material files
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filename, reader_config)) {
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
    // materials = reader.GetMaterials();

    //cout << "material name: " << materials[0].name << std::endl;
    // cout << materials.size() << std::endl;

    // Loop over shapes and load each attrib
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

#if USE_UV
                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                // don't texture yet
                if (idxa.texcoord_index >= 0) {
                    tinyobj::real_t txa = attrib.texcoords[2 * size_t(idxa.texcoord_index) + 0];
                    tinyobj::real_t tya = attrib.texcoords[2 * size_t(idxa.texcoord_index) + 1];
                    // vertA.hasUv = true;
                    vertA.uv = glm::vec2(txa, tya);
                }
                if (idxb.texcoord_index >= 0) {
                    tinyobj::real_t txb = attrib.texcoords[2 * size_t(idxb.texcoord_index) + 0];
                    tinyobj::real_t tyb = attrib.texcoords[2 * size_t(idxb.texcoord_index) + 1];
                    // vertB.hasUv = true;
                    vertB.uv = glm::vec2(txb, tyb);
                }
                if (idxc.texcoord_index >= 0) {
                    tinyobj::real_t txc = attrib.texcoords[2 * size_t(idxc.texcoord_index) + 0];
                    tinyobj::real_t tyc = attrib.texcoords[2 * size_t(idxc.texcoord_index) + 1];
                    // vertC.hasUv = true;
                    vertC.uv = glm::vec2(txc, tyc);
                }

                //printf("u: %f, v: %f \n", vertA.uv[0], vertA.uv[1]);
                //printf("u: %f, v: %f \n", vertB.uv[0], vertB.uv[1]);
                //printf("u: %f, v: %f \n", vertC.uv[0], vertC.uv[1]);
#endif
                // Optional: vertex colors
                // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];

                Triangle triangle = {
                    vertA,
                    vertB,
                    vertC
                };

                triangleArray->push_back(triangle);

            }
            index_offset += fv;

            // shapes[s].mesh.material_ids[f];
        }
    }

    return true;
}

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

        bool hasObj = false;
        string objFileName;

        //load object from obj file
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            }
            else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
            else if (strstr(line.c_str(), ".obj") != NULL) {
                cout << "Creating some obj..." << endl;
#if USE_BOUND_BOX
                newGeom.type = OBJ;
#else
                newGeom.type = TRIANGLE;
#endif

                hasObj = true;
                objFileName = line;
            }
            else {
                cout << "wtf is this??" << std::endl;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations and texture
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
            else if (strcmp(tokens[0].c_str(), "TEXTURE") == 0) {
                cout << "finding Texture... " << endl;
                newGeom.textureid = atof(tokens[1].c_str());
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        if (hasObj) {
            std::vector<Triangle> triangleArray;
            loadObj(objFileName.c_str(), &triangleArray);
#if USE_BOUND_BOX
            float xMin = FLT_MAX;
            float yMin = FLT_MAX;
            float zMin = FLT_MAX;
            float xMax = FLT_MIN;
            float yMax = FLT_MIN;
            float zMax = FLT_MIN;

            for (int i = 0; i < triangleArray.size(); i++) {
                // jank code to find the min and max of the box
                Triangle tri = triangleArray[i];

                xMin = fmin(fmin(tri.pointA.pos[0], tri.pointB.pos[0]), fmin(tri.pointC.pos[0], xMin));
                xMax = fmax(fmax(tri.pointA.pos[0], tri.pointB.pos[0]), fmax(tri.pointC.pos[0], xMax));

                yMin = fmin(fmin(tri.pointA.pos[1], tri.pointB.pos[1]), fmin(tri.pointC.pos[1], yMin));
                yMax = fmax(fmax(tri.pointA.pos[1], tri.pointB.pos[1]), fmax(tri.pointC.pos[1], yMax));

                zMin = fmin(fmin(tri.pointA.pos[2], tri.pointB.pos[2]), fmin(tri.pointC.pos[2], zMin));
                zMax = fmax(fmax(tri.pointA.pos[2], tri.pointB.pos[2]), fmax(tri.pointC.pos[2], zMax));
            }

            BoundBox box = {
                glm::vec3(xMin, yMin, zMin),
                glm::vec3(xMax, yMax, zMax)
            };

            cout << "xMin: " << xMin << " , " << yMin << " , " << zMin << endl;
            cout << "xMax: " << xMax << " , " << yMax << " , " << zMax << endl;


            newGeom.host_tris = new Triangle[triangleArray.size()];//triangleArray.size()];
            newGeom.device_tris = NULL;
            newGeom.numTris = triangleArray.size();

            newGeom.bound = box;
            for (int i = 0; i < triangleArray.size(); i++) {
                newGeom.host_tris[i] = triangleArray[i];
            }

            geoms.push_back(newGeom);

#else 
            // create geoms from triangles using newGeom properties
            // load triangles into the geoms scene.
            for (int i = 0; i < triangleArray.size(); i ++) {
                // there should only be 1 triangle
                Triangle* trisInGeom = new Triangle(triangleArray[i]);

                // just a single triangle
                Geom newTriGeom = {
                    TRIANGLE,
                    newGeom.materialid,
                    newGeom.translation,
                    newGeom.rotation,
                    newGeom.scale,
                    newGeom.transform,
                    newGeom.inverseTransform,
                    newGeom.invTranspose,
                    trisInGeom,
                    NULL, // device pointer is not yet allocated
                    BoundBox {
                        },
                    1,
                };
                geoms.push_back(newTriGeom);
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
    }
    else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            }
            else if (strcmp(line.c_str(), "cube") == 0) {
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
            }
            else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
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
    RenderState& state = this->state;
    Camera& camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "UP") == 0) {
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
    }
    else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.color = color;
            }
            else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            }
            else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

int Scene::loadTexture(string textureid) {
    int id = atoi(textureid.c_str());
    cout << "Loading Texture " << id << "..." << endl;
    Texture newTexture;
    glm::vec3* pixelData;
    int width, height, channels;

    if (id != textures.size()) {
        cout << "ERROR: TEXTURE ID does not match expected number of textures" << endl;
        return -1;
    }
    //load static properties
    for (int i = 0; i < 1; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);

        if (strcmp(tokens[0].c_str(), "PATH") == 0) {
            const char* filepath = tokens[1].c_str();
            unsigned char* data = stbi_load(filepath, &width, &height, &channels, 0);

            pixelData = new glm::vec3[width * height];

            // ... process data if not NULL ..
            if (data != nullptr && width > 0 && height > 0)
            {
                if (channels == 3)
                {
                    int pixelDataIdx = 0;
                    // iterate over every pixel
                    // total number of data points is width * height * channels (should be 3)
                    for (int p = 0; p < (width * height) * channels - 2; p += 3) {
                        glm::vec3 currPix = glm::vec3(static_cast<float>(data[p]) / 256.f,
                            static_cast<float>(data[p + 1]) / 256.f,
                            static_cast<float>(data[p + 2]) / 256.f);
                        pixelData[pixelDataIdx] = currPix;
                        
                        if (pixelDataIdx == 2300 || pixelDataIdx == width * height - 2300) {
                            cout << "pix: " << pixelDataIdx << " is: " << currPix[0] << ", " << currPix[1] << ", " << currPix[2] << endl;
                        }
                        pixelDataIdx++;
                    }

                    newTexture.width = width;
                    newTexture.height = height;
                    newTexture.host_texImage = pixelData;

                    cout << "Loaded all Texture Points" << endl;
                    cout << "width: " << newTexture.width; // looks good
                    cout << "height: " << newTexture.height; // looks good
                    cout << "last pixelIdx: " << pixelDataIdx; // looks correct
                }
            }
            else
            {
                std::cout << "Some error: channels are weird\n";
                return -1;
            }

            stbi_image_free(data);
        }
    }

    textures.push_back(newTexture);
    return 1;
}
