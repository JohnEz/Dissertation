#include "Branch.h"

Branch::Branch(int branchsize)
{
	srand(time(NULL));
	branchSize = branchsize;
	mesh = ExtendingMesh::GenerateExtendingCube(branchSize);
	mesh->SetTexture(SOIL_load_OGL_texture(TEXTUREDIR"stem.JPG", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, 0));


	modelScale = Vector3(2.5f,10.0f,2.5f);
	maxScale = Vector3(5, 10, 5);

	maxLeafs = 1;

	numLeafs = 0;

	sphere	= new OBJMesh(MESHDIR"ico.obj");
	hasPhysics = false;
	myNode = 0;
	endBranch = Vector3();
}

void Branch::Update(float msec)
{
	mesh->Update(msec);

	SceneNode::Update(msec);

	if (mesh->GetDrawIndices() == (branchSize * 24) && numLeafs < maxLeafs)
	{

		SceneNode* leaf = new SceneNode(sphere);

		float leafpos = (float)((mesh->GetDrawIndices() / 24) / 5);
		leaf->SetTransform(Matrix4::Translation(Vector3(0, leafpos * maxScale.y, 0)));
		leaf->SetModelScale(Vector3(20,20,20));

		AddChild(leaf);

		leaf->Update(msec);

		///////
		PhysicsNode*p = new PhysicsNode();

		float elements[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
		Matrix4 mat = Matrix4(elements);
		p->SetInverseInertia(mat);
		p->SetUseGravity(false);
		p->SetInverseMass(0.0f);

		p->SetPosition(leaf->GetWorldTransform().GetPositionVector());

		p->SetCollisionVolume(new CollisionLeaf(20));

		Vector3 startPos = GetWorldTransform().GetPositionVector();
		Vector3 endPos = leaf->GetWorldTransform().GetPositionVector();

		PhysicsSystem::GetPhysicsSystem().AddNode(p);

		/////////

		Light l = Light();

		l.SetPosition(leaf->GetWorldTransform().GetPositionVector());

		l.SetColour(Vector4 (0, 1, 0, 1.0f));

		l.SetRadius(150);

		LightStorage::GetInstance()->addLight(l);

		numLeafs++;

	}

	if (modelScale.x < maxScale.x)
	{
		modelScale.x += 0.000001;
		modelScale.z += 0.000001;
	}

	if (!hasPhysics && GetWorldTransform().GetPositionVector().Length() != 0)
	{

		SceneNode* leaf = new SceneNode(sphere);

		float leafpos = (float)((mesh->getNumInd() / 24) / 5);
		leaf->SetTransform(Matrix4::Translation(Vector3(0, leafpos * maxScale.y, 0)));

		AddChild(leaf);

		leaf->Update(msec);

		///////
		PhysicsNode*p = new PhysicsNode();

		float elements[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
		Matrix4 mat = Matrix4(elements);
		p->SetInverseInertia(mat);
		p->SetUseGravity(false);
		p->SetInverseMass(0.0f);

		p->SetPosition(leaf->GetWorldTransform().GetPositionVector());

		Vector3 startPos = GetWorldTransform().GetPositionVector();
		Vector3 endPos = leaf->GetWorldTransform().GetPositionVector();

		endBranch = endPos;

		p->SetCollisionVolume(new CollisionCylinder(1, startPos, endPos));

		myNode = p;

		PhysicsSystem::GetPhysicsSystem().AddNode(p);

		/////////

		hasPhysics = true;
		RemoveChild(leaf);
	}

	if (myNode)
	{
		float v = mesh->GetDrawIndices();
		float z = v / (branchSize * 24);
		Vector3 posDif = endBranch - GetWorldTransform().GetPositionVector();
		Vector3 endPos = GetWorldTransform().GetPositionVector() + (posDif * z);

		myNode->SetCollisionVolume(new CollisionCylinder(2, GetWorldTransform().GetPositionVector(), endPos));
	}

}