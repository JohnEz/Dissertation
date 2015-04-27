#pragma once
#include "ExtendingMesh.h"
#include ".\nclgl\SceneNode.h"
#include <time.h>
#include "MeshStorage.h"
#include "LightStorage.h"
#include "MyGame.h"
#include "GameEntity.h"

class Branch : public SceneNode
{
public:

	Branch(int branchsize);
	~Branch(void){};

	virtual void Update(float msec);

protected:

	int branchSize;
	int LeafAdd;
	int maxLeafs;
	int numLeafs;

	float counter;

	Vector3 maxScale;

	Mesh* sphere;

	PhysicsNode* myNode;
	Vector3 endBranch;

	bool hasPhysics;
};