#pragma once 
#include ".\nclgl\SceneNode.h"
#include <time.h>
#include "MeshStorage.h"
#include "PhysicsNode.h"
#include "Renderer.h"
#include "PhysicsSystem.h"

enum AgentState {
	PATROL,
	STARE_AT_PLAYER,
	CHASE_PLAYER,
	USE_ABILITY,
	MAX_STATES
};

enum HierarchicalState {
	DEFUALT,
	AGGROD
};

class Agent
{
public:

	const int PATROLSIZE = 3;

	Agent(SceneNode* s, PhysicsNode* p);
	~Agent(void){};

	void	Update(float msec);

	void	ConnectToSystems();
	void	DisconnectFromSystems();

protected:
	//FSM variables
	AgentState subState;
	HierarchicalState myState;
	int targetLocation;
	Vector3 patrolLocations[PATROLSIZE];

	//entity variables
	SceneNode*		renderNode;
	PhysicsNode*	physicsNode;

	bool removed;
};