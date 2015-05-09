#include "PhysicsNode.h"
#include <vector>

#ifndef AIMANAGER
#define AIMANAGER

enum State {
	PATROL,
	STARE_AT_PLAYER,
	CHASE_PLAYER,
	USE_ABILITY,
	LEASH,
	MAX_STATES
};

struct Ability {
	int damage;
	int maxCooldown;
	float cooldown;
	bool targetEnemy;
};

struct Players {

	static const int MAXPLAYERS = 5000;

	int level[MAXPLAYERS];
	int hp[MAXPLAYERS];
	int maxHP[MAXPLAYERS];

	bool isDead[MAXPLAYERS]; //
	
	float x[MAXPLAYERS];
	float y[MAXPLAYERS];
	float z[MAXPLAYERS];
	
};

struct AIWorldPartition {
	static const int MAXPARTITIONS = 100;
	Vector3 pos[MAXPARTITIONS];
	short* myPlayers;					//
	int playerCount[MAXPARTITIONS];		//
};

struct Agents {
	static const int MAXAGENTS = 1024 * 80 + 1;
	static const int AGGRORANGE = 1000;
	static const int MAXABILITIES = 3;

	State state[MAXAGENTS];
	int targetLocation[MAXAGENTS];
	Vector3 patrolLocation[MAXAGENTS][3];
	int targetPlayer[MAXAGENTS];
	Ability myAbilities[MAXAGENTS][MAXABILITIES];
	int level[MAXAGENTS];

	float x[MAXAGENTS];
	float y[MAXAGENTS];
	float z[MAXAGENTS];

	short* partitions;		//

};

struct CopyOnce
{
	Players myPlayers;
	Agents myAgents;
	AIWorldPartition myPartitions;
};

struct CopyEachFrame
{
	//players
	//bool isDead[MAXPLAYERS];
};

class AIManager {
public:
	//AIManager(){};
	//AIManager(int xNum, int yNum, int zNum, float height, float width, float depth);
	//~AIManager(){};

	static AIManager* GetInstance();

	void init(int xNum, int yNum, int zNum, float height, float width, float depth);

	void update(float msec);
	bool CheckBounding(PhysicsNode& n, float aggroRange,Vector3 pos, Vector3 halfDim);
	void Broadphase(float msec);
	void addAgent(PhysicsNode* a);
	void addPlayer(PhysicsNode* p);

	

protected:

	static AIManager* aiInst;

	vector<AIWorldPartition> allPartitions;
	Vector3 halfDim;

	Agents myAgents;
	Players myPlayers;
	AIWorldPartition myPartitions;

	int agentCount;
	int playerCount;
	int partitionCount;

	PhysicsNode* playerNodes[Players::MAXPLAYERS];
	PhysicsNode* agentNodes[Agents::MAXAGENTS];

	vector<int> myAgentsPlayers[Agents::MAXAGENTS];

	Ability agentAbilities[5];


};

#endif