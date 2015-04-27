#include "ExtendingMesh.h"


ExtendingMesh::ExtendingMesh(void)
{
	GrowthCounter = 0.0f;
}

ExtendingMesh::~ExtendingMesh(void)
{


}

void ExtendingMesh::Draw(bool update)
{
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, bumpTexture);

	glBindVertexArray(arrayObject);

	if (bufferObject[INDEX_BUFFER])
	{
		glDrawElements(type,drawIndices,GL_UNSIGNED_INT,0);
	}
	else
	{
		glDrawArrays(type, 0, (drawVertices));
	}

	glBindVertexArray(0);
}

void ExtendingMesh::ExtendMesh()
{
	if (bufferObject[INDEX_BUFFER])
	{
		if (drawIndices < numIndices)
		{
			drawIndices += (initialIndices);
			
		}
	}
	else
	{
		if (drawVertices < numVertices)
		{
			drawVertices += (initialVertices / 2);	
		}
	}
}

ExtendingMesh* ExtendingMesh::GenerateExtendingCube(int numberofextends)
{
	ExtendingMesh* m = new ExtendingMesh;
	m->numVertices = (8 + (4 * numberofextends));
	m->initialVertices = 8;
	m->drawVertices = 8;

	int maxExtends = numberofextends;
	int currentExtend = 0;


	m->vertices = new Vector3[m->numVertices];

	float y = 0.0f;
	for (int i = 0; i < (2 + numberofextends) * 4; i += 4)
	{
		//bottom
		m->vertices[i] = Vector3(-1.0f, y, 1.0f);
		m->vertices[i + 1] = Vector3(1.0f, y, 1.0f);
		m->vertices[i + 2] = Vector3(1.0f, y, -1.0f);
		m->vertices[i + 3] = Vector3(-1.0f, y, -1.0f);

		y += 0.2f;
	}

	m->textureCoords = new Vector2[m->numVertices];

	m->textureCoords[0] = Vector2(0.0f, 1.0f);
	m->textureCoords[1] = Vector2(1.0f, 1.0f);
	m->textureCoords[2] = Vector2(0.0f, 1.0f);
	m->textureCoords[3] = Vector2(1.0f, 1.0f);

	float texturespread = 0.1f;

	for (unsigned int i = 4; i < (m->numVertices); i += 4)
	{
		m->textureCoords[i] = Vector2(0.0f, 1 - texturespread);
		m->textureCoords[i + 1] = Vector2(1.0f, 1 - texturespread);
		m->textureCoords[i + 2] = Vector2(0.0f, 1 - texturespread);
		m->textureCoords[i + 3] = Vector2(1.0f, 1 - texturespread);

		texturespread += 0.1f;
	}

	m->initialIndices = 24;
	m->drawIndices = 24;
	m->numIndices = 24 + (24 * numberofextends);
	m->indices = new GLuint[m->numIndices];

	int count = 0;
	for (unsigned int i = 0; i < m->numIndices; i += 24)
	{
		int offset = 4 * count;

		//front 
		m->indices[i] = 4 + offset;
		m->indices[i + 1] = 0 + offset;
		m->indices[i + 2] = 1 + offset;
		m->indices[i + 3] = 1 + offset;
		m->indices[i + 4] = 5 + offset;
		m->indices[i + 5] = 4 + offset;

		//back
		m->indices[i + 6] = 6 + offset;
		m->indices[i + 7] = 2 + offset;
		m->indices[i + 8] = 3 + offset;
		m->indices[i + 9] = 3 + offset;
		m->indices[i + 10] = 7 + offset;
		m->indices[i + 11] = 6 + offset;

		//left
		m->indices[i + 12] = 7 + offset;
		m->indices[i + 13] = 3 + offset;
		m->indices[i + 14] = 0 + offset;
		m->indices[i + 15] = 0 + offset;
		m->indices[i + 16] = 4 + offset;
		m->indices[i + 17] = 7 + offset;

		//right
		m->indices[i + 18] = 5 + offset;
		m->indices[i + 19] = 1 + offset;
		m->indices[i + 20] = 2 + offset;
		m->indices[i + 21] = 2 + offset;
		m->indices[i + 22] = 6 + offset;
		m->indices[i + 23] = 5 + offset;

		count++;
	}

	m->colours = new Vector4[m->numVertices];
	for (unsigned int i = 0; i < m->numVertices; i++)
	{
		m->colours[i] = Vector4(1.0f, 1.0f, 1.0f, 1.0f);
	}

	m->GenerateNormals();
	m->GenerateTangents();

	m->BufferData();

	return m;
}

void ExtendingMesh::Update(float msec)
{
	GrowthCounter += msec / 100;

	if (GrowthCounter > 5.0f)
	{
		ExtendMesh();
		GrowthCounter -= 5.0f;
	}
}