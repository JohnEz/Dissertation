#pragma once 
#include ".\nclgl\SceneNode.h"
#include <time.h>
#include "MeshStorage.h"
#include "PhysicsNode.h"
#include "Renderer.h"
#include "PhysicsSystem.h"
#include "Player.h"

enum AgentState {
	PATROL,
	STARE_AT_PLAYER,
	CHASE_PLAYER,
	USE_ABILITY,
	LEASH,
	MAX_STATES
};

enum HierarchicalState {
	DEFUALT,
	AGGROD
};

struct Ability {
	float maxCooldown;
	float cooldown;
	int damage;
	bool targetEnemy;
};

class Agent
{
public:

	static const int PATROLSIZE = 3;
	static const int MAXABILITIES = 3;
	static const float MAXAGGRORANGE;

	Agent(SceneNode* s, PhysicsNode* p);
	~Agent(void){};

	void	Update(Player* players[], float msec);

	void	ConnectToSystems();
	void	DisconnectFromSystems();

//protected:
	//FSM variables
	AgentState subState;
	int targetLocation;
	Vector3* patrolLocations[PATROLSIZE];
	Player* targetPlayer;
	Ability* myAbilities[MAXABILITIES];
	int level;

	//entity variables
	SceneNode*		renderNode;
	PhysicsNode*	physicsNode;

	bool removed;
};