#include "Player.h"

const float MAXSPEED = 1.0f;

Player::Player(SceneNode* s, PhysicsNode* p) {
	renderNode	= s;
	physicsNode = p;
	removed = false;
}

void Player::Update(float msec)
{
}

void Player::ConnectToSystems() {
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

void Player::DisconnectFromSystems() {
	if(renderNode) {
		Renderer::GetRenderer().RemoveNode(renderNode);
	}

	if(physicsNode) {
		PhysicsSystem::GetPhysicsSystem().RemoveNode(physicsNode);
	}
}