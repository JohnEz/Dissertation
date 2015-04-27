#include "Stem.h"

Stem::Stem(PhysicsNode* myN)
{

	srand((unsigned int)time(NULL));
	stemSize = rand() % 50 + 50;
	stemSize = 100;
	mesh = ExtendingMesh::GenerateExtendingCube(stemSize);

	mesh->SetTexture(SOIL_load_OGL_texture(TEXTUREDIR"stem.jpg", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS));

	SetModelScale(Vector3(5, 10, 5));

	numBranches = 0;
	maxBranches = 8;

	maxLeafs = 1;
	numLeafs = 0;

	maxScale = Vector3(10, 10, 10);

	branchAdd = (((stemSize - (stemSize/maxBranches) )* 24) / maxBranches);

	myNode = myN;
}

void Stem::Update(float msec)
{

	mesh->Update(msec);

	SceneNode::Update(msec);

	if (mesh->GetDrawIndices() > branchAdd && numBranches < maxBranches)
	{
		int tempsize = (((rand() % (stemSize/2)) + stemSize/4));
		SceneNode* branch = new Branch(tempsize);

		float branchpos = (float)((branchAdd / 24) / 5);
		branchpos -= (branchpos / 9);
		
		branch->SetTransform(Matrix4::Translation(Vector3(0, branchpos * maxScale.y, 0)));


		float branchAngle = (float)((rand() % 180) - 90);
		while (branchAngle < 20 && branchAngle > -20)
		{
			branchAngle = (float)((rand() % 180) - 90);
		}
		branch->SetTransform(branch->GetTransform() * Matrix4::Rotation(branchAngle, Vector3(0, 0, 1)));
		branchAngle = (float)((rand() % 180) - 90);
		while (branchAngle < 20 && branchAngle > -20)
		{
			branchAngle = (float)((rand() % 180) - 90);
		}
		branch->SetTransform(branch->GetTransform() * Matrix4::Rotation(branchAngle, Vector3(1, 0, 0)));

		AddChild(branch);

		numBranches++;
		branchAdd += (((stemSize - stemSize / maxBranches) * 24) / maxBranches);

	}

	Vector3 endPos = myNode->GetPosition();
	float v = mesh->GetDrawIndices();
	float z = v / (stemSize * 24);
	endPos.y += 200 * z;

	myNode->SetCollisionVolume(new CollisionCylinder(2, myNode->GetPosition(), endPos));

	if (modelScale.x < maxScale.x)
	{
		modelScale.x += 0.0001;
		modelScale.z += 0.0001;
	}

		

}