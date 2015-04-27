#pragma once

#include "Matrix4.h"
#include <vector>
#include <cstring>
#include "Vector2.h"
#include "Shader.h"

class text2D {
public:
	void initText2D(const char* texturePath);
	void printText2D(const char * text, int x, int y, int size);
	void cleanupText2D();
protected:
	unsigned int Text2DTextureID;
	unsigned int Text2DVertexBufferID;
	unsigned int Text2DUVBufferID;
	Shader* Text2DShaderID;
	unsigned int Text2DUniformID;
};