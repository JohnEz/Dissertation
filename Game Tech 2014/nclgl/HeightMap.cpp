#include "HeightMap.h"

HeightMap::HeightMap(std::string name)
{
	std::ifstream file(name.c_str(),ios::binary);

	if (!file)
	{
		return;
	}

	numVertices = RAW_WIDTH * RAW_HEIGHT;
	numIndices = (RAW_WIDTH - 1) * (RAW_HEIGHT - 1) * 6;
	vertices = new Vector3[numVertices];
	textureCoords = new Vector2[numVertices];
	indices = new GLuint[numIndices];

	unsigned char* data = new unsigned char[numVertices];
	file.read((char*)data, numVertices*sizeof(unsigned char));
	file.close();

	for (int i = 0; i < RAW_WIDTH; i++)
	{
		for (int j = 0; j < RAW_HEIGHT; j++)
		{
			int offset = (i * RAW_WIDTH) + j;

			vertices[offset] = Vector3(i * HEIGHTMAP_X, data[offset] * HEIGHTMAP_Y, j * HEIGHTMAP_Z);

			textureCoords[offset] = Vector2(i * HEIGHTMAP_TEX_X, j * HEIGHTMAP_TEX_Z);
		}
	}

	delete data;

	numIndices = 0;

	for (int i = 0; i < RAW_WIDTH - 1; i++)
	{
		for (int j = 0; j < RAW_HEIGHT - 1; j++)
		{
			int a = (i * (RAW_WIDTH)) + j;
			int b = ((i + 1) * (RAW_WIDTH)) + j;
			int c = ((i + 1) * (RAW_WIDTH)) + (j + 1);
			int d = (i * (RAW_WIDTH)) + (j + 1);

			indices[numIndices++] = c;
			indices[numIndices++] = b;
			indices[numIndices++] = a;

			indices[numIndices++] = a;
			indices[numIndices++] = d;
			indices[numIndices++] = c;
		}
	}

	GenerateNormals();
	GenerateTangents();

	BufferData();

}