#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>


#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#if not STD_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include <stb_image.h>



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
            else if ((strcmp(tokens[0].c_str(), "SKYBOX") == 0)) {
                markSkyBox(tokens[1]);
            }
        }
    }
}

Scene::~Scene()
{
       
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
            else if (strcmp(line.c_str(), "object") == 0)
            {
                cout << "Creating new object geom..." << endl;
                newGeom.type = OBJECT;

                utilityCore::safeGetline(fp_in, line);
                vector<string> tokens = utilityCore::tokenizeString(line);
                if (loadGeomTriangles(newGeom, tokens[0]) != 1)
                {
                    cout << "ERROR: Load geom triangles fails" << endl;
                    return -1;
                }
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
            else if (strcmp(tokens[0].c_str(), "TEXTURE") == 0) {
                newMaterial.textureIndex = markTexture(tokens[1]);
                cout << "material bind to textureId: " << newMaterial.textureIndex << endl;
            }
            else if (strcmp(tokens[0].c_str(), "NORMALMAP") == 0) {
                newMaterial.normalMapIndex = markTextureNormal(tokens[1]);
                cout << "material bind to textureNormalId: " << newMaterial.normalMapIndex << endl;
            }
            else if (strcmp(tokens[0].c_str(), "METALLIC") == 0) {
                newMaterial.hasMetallic = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

int Scene::loadGeomTriangles(Geom& geom, string filePath)
{
    tinyobj::ObjReader reader;
    if (reader.ParseFromFile(filePath) == false)
    {
        std::cout << "tinyObj read error: " << reader.Error() << std::endl;
        return -1;
    }

    auto& attrs = reader.GetAttrib();
    auto& shapes = reader.GetShapes();

    geom.hasUV = attrs.texcoords.size() > 0;
    geom.hasNormal = attrs.normals.size() > 0;
    
    geom.triangleStartIndex = this->triangles.size();  
    for (int i = 0; i < shapes.size(); ++i)
    {
        Triangle t;
        int faceSize = shapes[i].mesh.material_ids.size();
        auto& indices = shapes[i].mesh.indices;
        for (int j = 0; j < faceSize; ++j)
        {
            int index0_v = indices[3 * j + 0].vertex_index;
            int index1_v = indices[3 * j + 1].vertex_index;
            int index2_v = indices[3 * j + 2].vertex_index;

            t.v0 = glm::vec3(attrs.vertices[3 * index0_v + 0], attrs.vertices[3 * index0_v + 1], attrs.vertices[3 * index0_v + 2]);
            t.v1 = glm::vec3(attrs.vertices[3 * index1_v + 0], attrs.vertices[3 * index1_v + 1], attrs.vertices[3 * index1_v + 2]);
            t.v2 = glm::vec3(attrs.vertices[3 * index2_v + 0], attrs.vertices[3 * index2_v + 1], attrs.vertices[3 * index2_v + 2]);

            if (geom.hasNormal)
            {
                int index0_n = indices[3 * j + 0].normal_index;
                int index1_n = indices[3 * j + 1].normal_index;
                int index2_n = indices[3 * j + 2].normal_index;

                t.n0 = glm::vec3(attrs.normals[3 * index0_n + 0], attrs.normals[3 * index0_n + 1], attrs.normals[3 * index0_n + 2]);
                t.n1 = glm::vec3(attrs.normals[3 * index1_n + 0], attrs.normals[3 * index1_n + 1], attrs.normals[3 * index1_n + 2]);
                t.n2 = glm::vec3(attrs.normals[3 * index2_n + 0], attrs.normals[3 * index2_n + 1], attrs.normals[3 * index2_n + 2]);
            }

            if (geom.hasUV)
            {
                int index0_tex = indices[3 * j + 0].texcoord_index;
                int index1_tex = indices[3 * j + 1].texcoord_index;
                int index2_tex = indices[3 * j + 2].texcoord_index;

                t.tex0 = glm::vec2(attrs.texcoords[2 * index0_tex + 0], attrs.texcoords[2 * index0_tex + 1]);
                t.tex1 = glm::vec2(attrs.texcoords[2 * index1_tex + 0], attrs.texcoords[2 * index1_tex + 1]);
                t.tex2 = glm::vec2(attrs.texcoords[2 * index2_tex + 0], attrs.texcoords[2 * index2_tex + 1]);
            }

#if ENABLE_NORMAL_MAP
            t.CacheTBN();
#endif 

            this->triangles.push_back(t);
        }
    }
    geom.triangleEndIndex = this->triangles.size();
    

#if ENABLE_BVH

    geom.bvhNodeStartIndex = bvhNodes.size();
    // Init PrimInfo vector
    std::vector<BVHPrimitiveInfo> bvhPrimInfos;
    std::vector<Triangle> reorderPrims;
    for (int i = geom.triangleStartIndex; i < geom.triangleEndIndex; ++i)
    {
        bvhPrimInfos.push_back(BVHPrimitiveInfo{ i, triangles[i].getBound(), triangles[i].getCentriod()});
    }

    // Build BVH
    int count = 0;
    BVHNode* root = recursiveBuildBVH(0, bvhPrimInfos.size(), count, bvhPrimInfos, reorderPrims);
    for (int i = geom.triangleStartIndex; i < geom.triangleEndIndex; ++i)
    {
        triangles[i] = reorderPrims[i - geom.triangleStartIndex];
    }
    
    // Transfer BVHNode to LinearBVHNode
    int offset = 0;
    std::vector<LinearBVHNode> tmpVec(count);
    flattenBVHNode(root, &offset, tmpVec);
    bvhNodes.insert(bvhNodes.end(), tmpVec.begin(), tmpVec.end());

    // Delete the BVHNode
    deleteBVHNode(root);

    geom.bvhNodeEndIndex = bvhNodes.size();

#endif

    return 1;
}

int Scene::markTexture(string filePath)
{
    for (int i = 0; i < textureIds.size(); ++i)
    {
        if (textureIds[i].compare(filePath) == 0)
        {
            return i;
        }
    }

    textureIds.emplace_back(filePath);

    return textureIds.size() - 1;
}

int Scene::markTextureNormal(string filePath)
{
    for (int i = 0; i < textureNormalIds.size(); ++i)
    {
        if (textureNormalIds[i].compare(filePath) == 0)
        {
            return i;
        }
    }

    textureNormalIds.emplace_back(filePath);

    return textureNormalIds.size() - 1;
}

int Scene::markSkyBox(string filePath)
{
    skyboxId = filePath;
    return 1;
}

BVHNode* Scene::recursiveBuildBVH(int st, int ed, int& count,
    std::vector<BVHPrimitiveInfo>& primInfos, std::vector<Triangle>& reorderPrims)
{
    int mid = (st + ed) / 2;        // Default split
    BVHNode* node = new BVHNode();
    count++;

    // Calc bound for current node
    Bound3 bound;
    for (int i = st; i < ed; ++i)
    {
        bound = bound.Union(primInfos[i].bound);
    }
    node->bound = bound;
    
    // If there is only one primitive, then no need to divide
    int nPrimitive = ed - st;
    if (nPrimitive <= 1)
    {      
        node->nPrimitives = nPrimitive;
        node->firstPrimOffset = reorderPrims.size();
        reorderPrims.push_back(triangles[primInfos[st].triangleId]);
        return node;
    }

    // Calc centriod bound and get split dim
    Bound3 centriodBound;
    for (int i = st; i < ed; ++i)
    {
        centriodBound = centriodBound.Union(primInfos[i].centroid);
    }
    int dim = centriodBound.MaxExtent();
    
    // If all primitive centriods are the same, then no need to divide
    if (centriodBound.pMax[dim] == centriodBound.pMin[dim])
    {
        node->nPrimitives = nPrimitive;
        node->firstPrimOffset = reorderPrims.size();
        for (int i = st; i < ed; ++i)
        {
            reorderPrims.push_back(triangles[primInfos[i].triangleId]);
        }
        return node;
    }
    
    // If there is only a few primitive, we can just sort them and divide in the middle
    if (nPrimitive <= 4)
    {
        std::nth_element(&primInfos[st], &primInfos[mid], &primInfos[ed - 1] + 1,
            [dim](const BVHPrimitiveInfo&a, const BVHPrimitiveInfo&b)
            {
                return a.centroid[dim] < b.centroid[dim];
            });
    }
    else
    {
        // SAH method to divide and reorder primitives

        // 1. Init bucket arrays and partition primitives into buckets by centriods
        const int nBucket = SAH_BUCKET_SIZE;
        struct BuckerInfo {
            int count = 0;
            Bound3 bounds;
        };
        BuckerInfo buckets[nBucket];
        for (int i = st; i < ed; ++i)
        {
            int b = nBucket * centriodBound.Offset(primInfos[i].centroid)[dim];
            if (b == nBucket) b = nBucket - 1;
            buckets[b].count++;
            buckets[b].bounds = buckets[b].bounds.Union(primInfos[i].bound);
        }

        // 2. Compute cost for splitting after each bucket
        const float primCost = 1.0f;    // The cost of intersection with a prim
        const float nodeCost = 0.125f;  // The cost of intersectino with the BVHNode
        float costs[nBucket - 1];
        for (int i = 0; i < nBucket - 1; ++i)
        {
            Bound3 b0;
            Bound3 b1;
            int count0 = 0, count1 = 0;
            for (int j = 0; j <= i; ++j)
            {
                b0 = b0.Union(buckets[j].bounds);
                count0 += buckets[j].count;
            }

            for (int j = i + 1; j < nBucket; ++j)
            {
                b1 = b1.Union(buckets[j].bounds);
                count1 +=  buckets[j].count;
            }

            costs[i] = nodeCost + 
                (count0 * primCost * b0.SurfaceArea() + count1 * primCost * b1.SurfaceArea()) / bound.SurfaceArea();
        }

        // 3. Find the minimum cost
        float minCost = costs[0];
        int minCostBucketIndex = 0;
        for (int i = 1; i < nBucket - 1; ++i)
        {
            if (costs[i] < minCost)
            {
                minCost = costs[i];
                minCostBucketIndex = i;
            }
        }

        // 4. Compare it with the cost of no division
        float leaveCost = primCost * nPrimitive;
        if (minCost < leaveCost || nPrimitive > MAX_PRIM_IN_BVH_NODE)
        {
            // a. When dividing by bucket yeilds min cost
            BVHPrimitiveInfo* pmid = std::partition(&primInfos[st], &primInfos[ed - 1] + 1,
                [=](const BVHPrimitiveInfo& p)
                {
                    int b = nBucket * centriodBound.Offset(p.centroid)[dim];
                    if (b == nBucket) b = nBucket - 1;
                    return b <= minCostBucketIndex;
                });

            mid = pmid - &primInfos[0];

            // No need to partition
            if (mid == st || mid == ed)
            {
                node->nPrimitives = nPrimitive;
                node->firstPrimOffset = reorderPrims.size();
                for (int i = st; i < ed; ++i)
                {
                    reorderPrims.push_back(triangles[primInfos[i].triangleId]);
                }
                return node;
            }
        }
        else
        {
            // b. When no division yeilds min cost
            node->nPrimitives = nPrimitive;
            node->firstPrimOffset = reorderPrims.size();
            for (int i = st; i < ed; ++i)
            {
                reorderPrims.push_back(triangles[primInfos[i].triangleId]);
            }
            return node;
        }
    }

    // 5. Construct interior node
    node->splitAxis = dim;
    node->leftChild = recursiveBuildBVH(st, mid, count, primInfos, reorderPrims);
    node->rightChild = recursiveBuildBVH(mid, ed, count, primInfos, reorderPrims);

    return node;
}

int Scene::flattenBVHNode(BVHNode* node, int *offset, std::vector<LinearBVHNode>& vec)
{
    LinearBVHNode* linearNode = &vec[*offset];
    linearNode->bound = node->bound;
    int myOffset = *offset;
    (*offset) = (*offset) + 1;

    if (node->nPrimitives > 0)  // Node is leaf node
    {
        linearNode->firstPrimOffset = node->firstPrimOffset;
        linearNode->nPrimitives = node->nPrimitives;
    }
    else    // Node is interior node
    {
        linearNode->axis = node->splitAxis;
        linearNode->nPrimitives = 0;

        // Left child is always the next element, so no need to store offset
        flattenBVHNode(node->leftChild, offset, vec);

        // Store right child offset
        linearNode->rightChildOffset = flattenBVHNode(node->rightChild, offset, vec);
    }

    return myOffset;
}

void Scene::deleteBVHNode(BVHNode* node)
{
    if (node->leftChild != nullptr) deleteBVHNode(node->leftChild);
    if (node->rightChild != nullptr) deleteBVHNode(node->rightChild);
    delete node;
}