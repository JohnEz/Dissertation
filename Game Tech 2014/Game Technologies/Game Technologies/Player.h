#pragma once 
#include ".\nclgl\SceneNode.h"
#include <time.h>
#include "MeshStorage.h"
#include "PhysicsNode.h"
#include "Renderer.h"
#include "PhysicsSystem.h"

class Player
{
public:
	static const int MAX_PLAYERS = 1000;

	int level;
	bool isDead;
	
	//entity variables
	SceneNode*		renderNode;
	PhysicsNode*	physicsNode;
	bool removed;

	Player(SceneNode* s, PhysicsNode* p);
	Player(){};
	~Player(void){};

	void	Update(float msec);

	void	ConnectToSystems();
	void	DisconnectFromSystems();

protected:

};