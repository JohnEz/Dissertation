#pragma once 
#include ".\nclgl\SceneNode.h"
#include "ExtendingMesh.h"
#include <time.h>
#include "Branch.h"
#include "MeshStorage.h"

class Stem : public SceneNode
{
public:

	Stem(PhysicsNode* myN);
	~Stem(void){};

	virtual void Update(float msec);

protected:


	int numBranches;
	int maxBranches;

	int stemSize;

	int branchAdd;

	int numLeafs;
	int maxLeafs;

	Vector3 maxScale;

	PhysicsNode* myNode;
};