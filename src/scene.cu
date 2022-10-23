#include "scene.h"
//#include <cuda.h>
//#include <cuda_runtime.h>


//int Scene::loadTexture(string filePath)
//{
//	for (int i = 0; i < textures.size(); ++i)
//	{
//		if (strcmp(textures[i].id, filePath.c_str()) == 0)
//		{
//			return i;
//		}
//	}
//
//	int width, height, channels;
//	stbi_ldr_to_hdr_gamma(1.0f);	// Disable gamma correction
//	float* img = stbi_loadf(filePath.c_str(), &width, &height, &channels, 3);
//
//	if (img == NULL)
//	{
//		cout << "Load texture [" << filePath << "] fails." << endl;
//		return -1;
//	}
//
//	Texture2D texture;
//	texture.id = filePath.c_str();
//	texture.width = width;
//	texture.height = height;
//	texture.channels = channels;
//
//	
//	// Set pixel data to device memory directly
//	//cudaMalloc((void**)&texture.dev_data, sizeof(glm::vec3) * (width * height));
//	//checkCUDAError2("cudaMalloc texture error");
//	//cudaMemcpy(texture.dev_data, img, sizeof(float) * (width * height * 3), cudaMemcpyHostToDevice);
//	//checkCUDAError2("cudaMemcpy texture error");
//	//cudaDeviceSynchronize();
//
//	textures.push_back(texture);
//
//	stbi_image_free(img);
//
//	cout << "load texture done" << endl;
//
//	return textures.size() - 1;
//}

