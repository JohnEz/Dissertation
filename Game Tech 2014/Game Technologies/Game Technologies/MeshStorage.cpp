#include "MeshStorage.h"

//sets instance pointer to null
MeshStorage* MeshStorage::pInst = NULL;

Mesh* MeshStorage::berry = NULL;
Mesh* MeshStorage::leaf = NULL;


MeshStorage::MeshStorage()
{
	//CreateLeaf();
	//CreateBerry();
}

MeshStorage::~MeshStorage()
{

}

//returns instance pointer
MeshStorage* MeshStorage::GetInstance()
{
	if (pInst == NULL)
	{
		pInst = new MeshStorage();
	}

	return pInst;
}

Mesh* MeshStorage::GetStoredMesh(StoredMeshes mesh)
{
	switch (mesh)
	{
	case LEAF: { return leaf; }
		break;
	case BERRY: {return berry; }
		break;
	default: {return NULL; }
		break;
	}
}