#include "GameEntity.h"
#include "Renderer.h"
#include "PhysicsSystem.h"

GameEntity::GameEntity(void)	{
	renderNode	= NULL;
	physicsNode = NULL;
	removed = false;
}

GameEntity::GameEntity(SceneNode* s, PhysicsNode* p) {
	renderNode	= s;
	physicsNode = p;
	removed = false;
}

GameEntity::~GameEntity(void)	{
	DisconnectFromSystems();

	delete renderNode;
	delete physicsNode;
}

void	GameEntity::Update(float msec) {
	if (physicsNode->GetMaxCol() != -1 && physicsNode->GetMaxCol() < physicsNode->GetCurrentCol() && !removed)
	{
		DisconnectFromSystems();
		removed = true;
		BuildSphereEntity(5.0f, physicsNode->GetPosition() + Vector3(10, 0, 0), Vector3(0.5f, 0, 0));
		BuildSphereEntity(5.0f, physicsNode->GetPosition() + Vector3(-10, 0, 0), Vector3(-0.5f, 0, 0));
		BuildSphereEntity(5.0f, physicsNode->GetPosition() + Vector3(0, 10, 0), Vector3(0, 1, 0));
		BuildSphereEntity(5.0f, physicsNode->GetPosition() + Vector3(0, 0, 10), Vector3(0, 0, 0.5f));
		BuildSphereEntity(5.0f, physicsNode->GetPosition() + Vector3(0, 0, -10), Vector3(0, 0, -0.5f));

		Performance::GetInstance()->addScore(100);
	}
}

/*
Makes a sphere.
*/
GameEntity* GameEntity::BuildSphereEntity(float radius, Vector3 pos, Vector3 vel) {
	SceneNode* s = new SceneNode(renderNode->GetMesh());

	s->SetModelScale(Vector3(radius,radius,radius));
	s->SetBoundingRadius(radius);
	s->SetColour(Vector4(0,0,1,1));

	PhysicsNode*p = new PhysicsNode();
	p->SetPosition(pos);
	p->SetLinearVelocity(vel);
	p->SetAngularVelocity(Vector3(0, 0, 0));

	float I = 2.5f/(1.0f*radius*radius);
	float elements[] = {I, 0, 0, 0, 0, I, 0, 0, 0, 0, I, 0, 0, 0, 0, 1};
	Matrix4 mat = Matrix4(elements);
	p->SetInverseInertia(mat);

	p->SetInverseMass(1.0f);

	p->SetCollisionVolume(new CollisionSphere(radius));

	GameEntity*g = new GameEntity(s, p);
	g->ConnectToSystems();
	return g;
}

void	GameEntity::ConnectToSystems() {
	if(renderNode) {
		Renderer::GetRenderer().AddNode(renderNode);
	}

	if(physicsNode) {
		PhysicsSystem::GetPhysicsSystem().AddNode(physicsNode);
	}

	if(renderNode && physicsNode) {
		physicsNode->SetTarget(renderNode);
	}
}

void	GameEntity::DisconnectFromSystems() {
	if(renderNode) {
		Renderer::GetRenderer().RemoveNode(renderNode);
	}

	if(physicsNode) {
		PhysicsSystem::GetPhysicsSystem().RemoveNode(physicsNode);
	}
}