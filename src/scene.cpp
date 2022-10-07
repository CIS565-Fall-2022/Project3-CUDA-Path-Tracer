#include "scene.h"

#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/epsilon.hpp>

#include "TinyObjLoader/tiny_obj_loader.h"
#include "stb_image.h"
#include "ColorConsole/color.hpp"
#include "consts.h"

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

// if load_render_state flag is true, then the camera & renderstate will be initialized from the file
// otherwise they must be initialized manually
Scene::Scene(std::string filename, bool load_render_state) : filename(filename) {
    std::cout << "Reading scene from " << filename << " ..." << std::endl;
    std::cout << " " << std::endl;

    fp_in.open(filename);
    if (!fp_in.is_open()) {
        std::cerr << dye::red("Error reading from file - aborting!") << std::endl;
        throw;
    }

    // saves each geom's min & max vertices
    std::unordered_map<int, std::pair<glm::vec3, glm::vec3>> geom_id_to_extremes;
    int attrib_flags = 0;
    glm::vec3 world_min(FLT_MAX), world_max(FLT_MIN);
    while (fp_in.good()) {
        std::string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            std::vector<std::string> tokens = utilityCore::tokenizeString(line);
            if (tokens[0] == "OBJECT") {
                attrib_flags |= 1;
                if (!loadGeom()) {
                    std::cerr << dye::red("Error Loading Geoms") << std::endl;
                    throw;
                } else {
                    world_min = glm::min(world_min, geoms.back().bounds.min());
                    world_max = glm::max(world_max, geoms.back().bounds.max());
                }
            } else if (tokens[0] == "CAMERA" && load_render_state) {
                attrib_flags |= 1 << 1;
                loadCamera();
            }
        }
    }

    if (load_render_state && attrib_flags != 3) {
        std::cerr << dye::red("Scene " + filename + " is Malformed") << std::endl;
        throw;
    }

    // calculate world AABB
    world_AABB = AABB(world_min, world_max);
}

Scene::~Scene() {

}

static bool initMaterial(
    Scene& self,
    Material& ret,
    std::string const& obj_dir,
    tinyobj::material_t const& tinyobj_mat
) {
    Material mat;
    mat.diffuse = color_t(tinyobj_mat.diffuse[0], tinyobj_mat.diffuse[1], tinyobj_mat.diffuse[2]);
    mat.specular.color = color_t(tinyobj_mat.specular[0], tinyobj_mat.specular[1], tinyobj_mat.specular[2]);
    mat.specular.exponent = tinyobj_mat.shininess;
    
    auto& params = tinyobj_mat.unknown_parameter;

#define PARSE_F(name_str, field, default_val)\
    do {\
    if (params.count(name_str)) field = std::stof(params.find(name_str)->second);\
    else field = default_val;\
    } while(0)

    PARSE_F("reflect", mat.hasReflective, 0);
    PARSE_F("refr", mat.hasRefractive, 0);
    PARSE_F("ior", mat.ior, 1);
    PARSE_F("emit", mat.emittance, 0);
    PARSE_F("rough", mat.roughness, 0);

#undef PARSE_F

    auto load_texture = [&](int& id, std::string const& texname) {
        if (!texname.empty()) {
            if (self.tex_name_to_id.count(texname)) {
                mat.textures.diffuse = self.tex_name_to_id[texname];
            } else {
                int x, y, n;
                std::string texpath = obj_dir + '/' + texname;

                unsigned char* data = stbi_load(texpath.c_str(), &x, &y, &n, NUM_TEX_CHANNEL);
                if (!data) {
                    return false;
                }

                self.textures.emplace_back(x, y, data);
                stbi_image_free(data);

                id = self.tex_name_to_id[texname] = self.textures.size() - 1;
            }
        } else {
            id = -1;
        }

        return true;
    };

    if (!load_texture(mat.textures.diffuse, tinyobj_mat.diffuse_texname)) {
        return false;
    }
    if (!load_texture(mat.textures.bump, tinyobj_mat.bump_texname)) {
        return false;
    }

    // TODO deduce material type
#ifdef TRANSPARENT
#undef TRANSPARENT
#endif

    if(mat.hasReflective > 0 && mat.hasRefractive > 0) {
        mat.type = Material::Type::GLOSSY;
    } else if (mat.hasReflective > 0) {
        mat.type = Material::Type::REFL;
    } else if (mat.hasRefractive > 0) {
        if (abs(mat.hasRefractive - 1) <= EPSILON) {
            mat.type = Material::Type::TRANSPARENT;
        } else {
            mat.type = Material::Type::REFR;
        }
    } else {
        mat.type = Material::Type::DIFFUSE;
    }

    if (mat.type == Material::Type::DIFFUSE || mat.type == Material::Type::GLOSSY) {
        if (mat.roughness <= EPSILON && mat.roughness >= -EPSILON) {
            mat.roughness = 1; // force roughness for diffuse and glossy material
        }
    }

    std::cout << "loaded material " << tinyobj_mat.name << std::endl
        << "diffuse =     {" << mat.diffuse[0] << "," << mat.diffuse[1] << "," << mat.diffuse[2] << "}\n"
        << "emittance =    " << mat.emittance << "\n"
        << "ior =          " << mat.ior << "\n"
        << "refl =         " << mat.hasReflective << "\n"
        << "refr =         " << mat.hasRefractive << "\n"
        << "roughness =    " << mat.roughness << "\n"
        << "spec_color=   {" << mat.specular.color[0] << "," << mat.specular.color[1] << "," << mat.specular.color[2] << "}\n"
        << "spec_exp   =   " << mat.specular.exponent << "\n"
        << "type =         " << mat.type << "\n\n";

    if(mat.textures.diffuse != -1)
        std::cout << "diffuse tex = {" << " id = " << mat.textures.diffuse << ", npixels = " << self.textures[mat.textures.diffuse].pixels.size() << "}\n";
    if(mat.textures.bump != -1)
        std::cout << "bump tex    = {" << " id = " << mat.textures.bump << ", npixels = " << self.textures[mat.textures.bump].pixels.size() << "}\n";


    ret = std::move(mat);
    return true;
}

bool Scene::loadGeom() {
    int objectid = geoms.size();
    std::cout << "Loading Geom " << objectid << "..." << std::endl;
    Geom newGeom;
    std::string line;

    //load object type
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        if (line == "sphere") {
            std::cout << "Creating new sphere..." << std::endl;
            newGeom.type = SPHERE;
        } else if (line ==  "cube") {
            std::cout << "Creating new cube..." << std::endl;
            newGeom.type = CUBE;
        } else {
            newGeom.type = MESH;

            // mesh objects are in the fomat: [file type] [path to file]
            std::vector<std::string> tokens = utilityCore::tokenizeString(line);
            if (tokens.size() != 2) {
                std::cerr << dye::red("ERROR: unrecognized object type\nat line: ") << line << std::endl;
                return false;
            }
            if (tokens[0] == "obj") {
                std::cout << "Loading obj mesh " << tokens[1] << std::endl;
                size_t pos = tokens[1].find_last_of('/');
                if (pos == std::string::npos) {
                    std::cerr << dye::red("ERROR: invalid obj file path: " + tokens[1]) << std::endl;
                    return false;
                }
                
                // default material folder to the folder where the mesh is in
                tinyobj::ObjReaderConfig config;
                config.mtl_search_path = tokens[1].substr(0, pos);
                std::cout << "set material lookup path to: " << config.mtl_search_path << std::endl;
                

                tinyobj::ObjReader reader;
                if (!reader.ParseFromFile(tokens[1], config)) {
                    if (!reader.Error().empty()) {
                        std::cerr << dye::red("TinyObjReader: ERROR: \n");
                        std::cerr << dye::red(reader.Error()) << std::endl;
                    } else {
                        std::cerr << dye::red("no idea what the hell is happening\n");
                    }
                    return false;
                }
                if (!reader.Warning().empty()) {
                    std::cerr << dye::yellow("TinyObjReader: WARNING: \n");
                    std::cerr << dye::yellow(reader.Warning()) << std::endl;
                }

                size_t vert_offset = vertices.size();
                size_t norm_offset = normals.size();
                size_t uv_offset = uvs.size();
                size_t mat_offset = materials.size();
                size_t tan_offset = tangents.size();

                auto const& attrib = reader.GetAttrib();
                auto const& mats = reader.GetMaterials();

                bool has_normal_map = false;
                
                // fill materials
                for (auto const& mat : mats) {
                    Material material;
                    if (!initMaterial(*this, material, config.mtl_search_path, mat)) {
                        std::cerr << dye::red("FATAL ERROR: mesh " + tokens[1] + " is missing some texture files:\n" +
                           mat.diffuse_texname + "\nresulting in an incomplete state of the loader, exiting...\n");
                        exit(EXIT_FAILURE);
                        return -1;
                    }
                    if (material.textures.bump != -1) {
                        has_normal_map = true;
                    }
                    materials.emplace_back(material);
                }

                // fill vertices
                for (size_t i = 2; i < attrib.vertices.size(); i += 3) {
                    vertices.emplace_back(
                        attrib.vertices[i - 2],
                        attrib.vertices[i - 1],
                        attrib.vertices[i]
                    );
                }
                // fill normals
                for (size_t i = 2; i < attrib.normals.size(); i += 3) {
                    normals.emplace_back(
                        attrib.normals[i - 2],
                        attrib.normals[i - 1],
                        attrib.normals[i]
                    );
                }
                // fill uvs
                for (size_t i = 1; i < attrib.texcoords.size(); i += 2) {
                    uvs.emplace_back(
                        attrib.texcoords[i - 1],
                        attrib.texcoords[i]
                    );
                }

                // temp buffer to compute tangent and bitangent
                int num_verts = attrib.vertices.size() / 3;
                std::vector<glm::vec3> tan1(num_verts, glm::vec3(0));
                std::vector<glm::vec3> tan2(num_verts, glm::vec3(0));
                std::vector<glm::vec3> vert2norm(num_verts);

                // maps normal to normal buffer index (used only if the model is missing normals)
                auto h = [](glm::vec3 const& v)->size_t {
                    auto hs = std::hash<float>();
                    return hs(v.x) ^ hs(v.y) ^ hs(v.z);
                };
                auto eq = [](glm::vec3 const& v1, glm::vec3 const& v2)->bool {
                    for (int i = 0; i < 3; ++i) {
                        if (!glm::epsilonEqual(v1[i], v2[i], 0.001f)) {
                            return false;
                        }
                    }
                    return true;
                };
                std::unordered_map<glm::vec3,int,decltype(h),decltype(eq)> normal_deduction_mp(10,h,eq);
                auto add_or_get_norm_id = [&](glm::vec3 const& v) {
                    if (normal_deduction_mp.count(v)) {
                        return normal_deduction_mp[v];
                    } else {
                        int ret = normal_deduction_mp[v] = normals.size();
                        normals.emplace_back(v);
                        return ret;
                    }
                };

                auto compute_tan1tan2 = [&](glm::ivec3 iverts, glm::ivec3 iuvs) {
                    // reference: Lengyel, Eric. "Computing Tangent Space Basis std::vectors for an Arbitrary Mesh."
                    // Terathon Software 3D Graphics Library, 2001. http://www.terathon.com/code/tangent.html
                    int i1 = iverts[0], i2 = iverts[1], i3 = iverts[2];
                    glm::vec3 v1 = vertices[iverts[1]] - vertices[iverts[0]];
                    glm::vec3 v2 = vertices[iverts[2]] - vertices[iverts[0]];
                    glm::vec2 u1 = uvs[iuvs[1]] - uvs[iuvs[0]];
                    glm::vec2 u2 = uvs[iuvs[2]] - uvs[iuvs[0]];
                    float f = 1.0f / (u1.x * u2.y - u2.x * u1.y);
                    glm::vec3 sd = (v1 * u2.y - v2 * u1.y) * f;
                    glm::vec3 td = (v2 * u1.x - v1 * u2.x) * f;

                    for (int i = 0; i < 3; ++i) {
                        tan1[iverts[i] - vert_offset] += sd;
                        tan2[iverts[i] - vert_offset] += td;
                    }
                };

                int triangles_start = triangles.size();
                bool missing_norm = false;
                bool missing_uv = false;
                for (auto const& s : reader.GetShapes()) {
                    auto const& indices = s.mesh.indices;
                    for (size_t i = 0; i < s.mesh.material_ids.size(); ++i) {
                        glm::ivec3 verts {
                            indices[3 * i + 0].vertex_index + vert_offset,
                            indices[3 * i + 1].vertex_index + vert_offset,
                            indices[3 * i + 2].vertex_index + vert_offset,
                        };

                        glm::ivec3 norms, uvs;
                        for (int x = 0; x < 3; ++x) {
                            if (indices[3 * i + x].normal_index == -1) {
                                missing_norm = true;
                                norms[x] = -1;
                            } else {
                                norms[x] = indices[3 * i + x].normal_index + norm_offset;
                            }

                            if (indices[3 * i + x].texcoord_index == -1) {
                                missing_uv = true;
                                uvs[x] = -1;
                            } else {
                                uvs[x] = indices[3 * i + x].texcoord_index + uv_offset;
                            }
                        }

                        if (missing_norm) {
                            // deduce normal from cross product
                            glm::vec3 v0v1 = vertices[verts[1]] - vertices[verts[0]];
                            glm::vec3 v0v2 = vertices[verts[2]] - vertices[verts[0]];
                            norms[0] = norms[1] = norms[2] = add_or_get_norm_id(glm::cross(v0v1, v0v2));
                        }

                        int mat_id;
                        if (s.mesh.material_ids[i] >= 0) {
                            mat_id = s.mesh.material_ids[i] + mat_offset;
                        } else {
                            mat_id = -1;
                        }
                        
                        // compute temp buffers for tangent computation
                        if(has_normal_map && !missing_uv) {
                            compute_tan1tan2(verts, uvs);
                            vert2norm[verts[0] - vert_offset] = normals[norms[0]];
                            vert2norm[verts[1] - vert_offset] = normals[norms[1]];
                            vert2norm[verts[2] - vert_offset] = normals[norms[2]];
                        }

                        triangles.emplace_back(verts, norms, uvs, mat_id);
                    }
                }

                if (has_normal_map && !missing_uv) {
                    for (size_t i = 0; i < num_verts; ++i) {
                        Normal const& n = vert2norm[i];
                        glm::vec3 const& t = tan1[i];
                        glm::vec3 const& t2 = tan2[i];
                        // Gram-Schmidt orthogonalize
                        // the 4th component stores handedness
                        tangents.emplace_back(glm::vec4(
                            glm::normalize((t - n * glm::dot(n, t))),
                            glm::dot(glm::cross(n, t), t2) < 0 ? -1.0f : 1.0f
                        ));
                    }

                    auto triangle_it = triangles.begin() + triangles_start;
                    for (auto const& s : reader.GetShapes()) {
                        auto const& indices = s.mesh.indices;
                        for (size_t i = 0; i < s.mesh.material_ids.size(); ++i) {
                            // tangent indices are just verts[0], verts[1], verts[2]
                            // because size of tangent buffer is the num of verts
                            (triangle_it++)->tangents = glm::ivec3(
                                indices[3 * i + 0].vertex_index + tan_offset,
                                indices[3 * i + 1].vertex_index + tan_offset,
                                indices[3 * i + 2].vertex_index + tan_offset
                            );
                        }
                    }
                }

                newGeom.meshid = meshes.size();
                meshes.emplace_back(triangles_start, triangles.size());

                

                std::cout << dye::green("Loaded:\n")
                    << triangles.size() << " triangles\n"
                    << vertices.size() << " vertices\n"
                    << normals.size() << " normals\n"
                    << uvs.size() << " uvs\n"
                    << meshes.size() << " meshes\n"
                    << tangents.size() << " tangents\n";

                if (missing_uv) {
                    std::cout << dye::red("missing uv for " + tokens[1]) << std::endl;
                }
                if (missing_norm) {
                    std::cout << dye::red("missing norm for " + tokens[1]) << std::endl;
                }

            } else {
                std::cerr << "unknown object format" << std::endl;
                return -1;
            }
        }
    }
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (tokens[0] == "material") {
            int mat_id = loadMaterial(tokens[1]);
            if (mat_id < 0) {
                return false;
            }
            newGeom.materialid = mat_id;
            std::cout << "Connecting Geom " << objectid << " to Material " << tokens[1] << "..." << std::endl;
        } else {
            std::cerr << "unknown field: " << tokens[0] << std::endl;
            return false;
        }
    }
    //load transformations
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (tokens[0] == "TRANS") {
            newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (tokens[0] == "ROTAT") {
            newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (tokens[0] == "SCALE") {
            newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        utilityCore::safeGetline(fp_in, line);
    }

    newGeom.transform = utilityCore::buildTransformationMatrix(
        newGeom.translation, newGeom.rotation, newGeom.scale);
    newGeom.inverseTransform = glm::inverse(newGeom.transform);
    newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

    // record lights, for now primitives only
    if (newGeom.type != MESH) {
        Material const& mat = materials[newGeom.materialid];
        if (mat.emittance > 0) {
            Light light;
            light.color = mat.diffuse;
            light.intensity = mat.emittance / MAX_EMITTANCE;
            light.position = newGeom.translation;
            lights.emplace_back(light);
        }
    }
    // compute AABB
    glm::vec3 geom_min = glm::vec3(FLT_MAX), geom_max = glm::vec3(FLT_MIN);
    if (newGeom.type == CUBE) {
        constexpr float vals[] = { -PRIM_CUBE_EXTENT, PRIM_CUBE_EXTENT };
        for (float x : vals) {
            for (float y : vals) {
                for (float z : vals) {
                    glm::vec3 vert = glm::vec3(newGeom.transform * glm::vec4(x, y, z, 1));
                    geom_min = glm::min(geom_min, vert);
                    geom_max = glm::max(geom_max, vert);
                }
            }
        }
    } else if (newGeom.type == SPHERE) {
        glm::vec3 center = glm::vec3(newGeom.transform * glm::vec4(0, 0, 0, 1));
        geom_min = center - glm::vec3(PRIM_SPHERE_RADIUS) * newGeom.scale;
        geom_max = center + glm::vec3(PRIM_SPHERE_RADIUS) * newGeom.scale;
    } else if (newGeom.type == MESH) {
        for (int i = meshes[newGeom.meshid].tri_start; i < meshes[newGeom.meshid].tri_end; ++i) {
            glm::vec3 verts[] = {
                vertices[triangles[i].verts[0]],
                vertices[triangles[i].verts[1]],
                vertices[triangles[i].verts[2]]
            };
            for (glm::vec3 const& vert : verts) {
                glm::vec3 world_vert = glm::vec3(newGeom.transform * glm::vec4(vert, 1));
                geom_min = glm::min(geom_min, world_vert);
                geom_max = glm::max(geom_max, world_vert);
            }
        }
    } else {
        std::cerr << dye::red("WTF?\n");
        exit(77777);
    }
    newGeom.bounds = AABB(geom_min, geom_max);
    geoms.push_back(newGeom);
    return true;
}

void Scene::loadCamera() {
    std::cout << "Loading Camera ..." << std::endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        std::string line;
        utilityCore::safeGetline(fp_in, line);
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (tokens[0] == "RES") {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (tokens[0] == "FOVY") {
            fovy = atof(tokens[1].c_str());
        } else if (tokens[0] == "ITERATIONS") {
            state.iterations = atoi(tokens[1].c_str());
        } else if (tokens[0] == "DEPTH") {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (tokens[0] == "FILE") {
            state.imageName = tokens[1];
        }
    }

    std::string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (tokens[0] == "EYE") {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (tokens[0] == "LOOKAT") {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (tokens[0] == "UP") {
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

    std::cout << "Loaded camera!" << std::endl;
}


// standardize material using mtl format
// returns the material id on success or -1 on error
// NOTE: only loads 1 material per mtl file, the rest will be discarded
int Scene::loadMaterial(std::string mtl_file) {    
    if (mtl_to_id.count(mtl_file)) {
        return mtl_to_id[mtl_file];
    } else {
        std::ifstream fin(mtl_file);
        if (!fin) {
            std::cerr << dye::red("cannot find mtl file: ") << dye::red(mtl_file) << std::endl;
            return -1;
        }

        std::map<std::string, int> mat_mp;
        std::vector<tinyobj::material_t> mats;
        std::string warn, err;

        tinyobj::LoadMtl(&mat_mp, &mats, &fin, &warn, &err);
        if (!err.empty()) {
            std::cerr << dye::red("Tiny obj loader: ERROR:\n") << dye::red(err) << std::endl;
            return -1;
        }
        if (!warn.empty()) {
            std::cerr << dye::yellow("Tiny obj loader: WARNING:\n") << dye::yellow(warn) << std::endl;
        }

        if (mat_mp.size() > 1) {
            std::cerr << dye::yellow("WARNING: ") << dye::yellow(mat_mp.size()-1) << dye::yellow("materials discarded in ") << mtl_file << std::endl;
        }

        Material material;
        if (!initMaterial(*this, material, mtl_file.substr(0, mtl_file.find_last_of('/')), mats[0])) {
            return -1;
        }
        materials.emplace_back(material);
        
        mtl_to_id[mtl_file] = materials.size() - 1;
        return materials.size() - 1;
    }
}