#include "text2D.h"

void text2D::initText2D(const char* texturePath){

	// Initialize texture
	Text2DTextureID = SOIL_load_OGL_texture(texturePath, SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, 0);

	// Initialize VBO
	glGenBuffers(1, &Text2DVertexBufferID);
	glGenBuffers(1, &Text2DUVBufferID);

	// Initialize Shader
	Text2DShaderID = new Shader(SHADERDIR"TextVertex.glsl", SHADERDIR"TextFrag.glsl");

	// Initialize uniforms' IDs
	Text2DUniformID = glGetUniformLocation(Text2DShaderID->GetProgram(), "myTextureSampler" );

}

void text2D::printText2D(const char* text, int x, int y, int size)
{

}



void text2D::cleanupText2D(){

	// Delete buffers
	glDeleteBuffers(1, &Text2DVertexBufferID);
	glDeleteBuffers(1, &Text2DUVBufferID);

	// Delete texture
	glDeleteTextures(1, &Text2DTextureID);

	// Delete shader
	glDeleteProgram(Text2DShaderID->GetProgram());
}
