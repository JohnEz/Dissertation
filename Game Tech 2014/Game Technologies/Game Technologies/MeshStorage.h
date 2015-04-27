#pragma once
#include ".\nclgl\OBJMesh.h"

enum StoredMeshes{LEAF, BERRY};

class MeshStorage
{
public:

	//returns pointer to worldmap instance
	static MeshStorage* GetInstance();

	Mesh* GetStoredMesh(StoredMeshes mesh);

protected:

	static Mesh* leaf;
	static Mesh* berry;

	MeshStorage();
	~MeshStorage();

	//static void CreateBerry()
	//{
	//	OBJMesh* m = new OBJMesh();
	//	//m->LoadOBJMesh(MESHDIR"sphere.obj", TEXTUREDIR"Bumpy Textured Leaves.jpg");
	//	berry = m;
	//}

	//pointer to worldmap instance
	static MeshStorage* pInst;

};