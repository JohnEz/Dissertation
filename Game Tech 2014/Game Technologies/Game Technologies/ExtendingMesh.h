#pragma once

#include"./nclgl/Mesh.h"

class ExtendingMesh: public Mesh
{
public:

	ExtendingMesh(void);
	~ExtendingMesh(void);

	virtual void Draw(bool update = true);

	virtual void Update(float msec);

	void ExtendMesh();

	static ExtendingMesh* GenerateExtendingCube(int numberofextends);

	virtual int GetDrawIndices(){ return drawIndices; }

protected:

	int initialVertices;
	int initialIndices;
	int drawVertices;
	int drawIndices;

	float GrowthCounter;
};