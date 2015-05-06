#include "Agent.h"
#include "Player.h"
#include "PhysicsNode.h"
#include <vector>

#ifndef AIMANAGER
#define AIMANAGER

/*struct AIWorldPartition{
	Vector3 pos;
	vector<Agent*> myAgents;
	vector<Player*> myPlayers;
};*/

struct AIWorldPartition{
	Vector3 pos;
	vector<int> myPlayers;
};

enum State {
	PATROLS,
	STARE_AT_PLAYERS,
	CHASE_PLAYERS,
	USE_ABILITYS,
	LEASHS,
	MAX_STATESS
};

struct Players {

	static const int MAXPLAYERS = 100;

	int level[MAXPLAYERS];
	int hp[MAXPLAYERS];
	int maxHP[MAXPLAYERS];
	bool isDead[MAXPLAYERS];
	
	float x[MAXPLAYERS];
	float y[MAXPLAYERS];
	float z[MAXPLAYERS];
	
};

struct Agents {
	static const int MAXAGENTS = 4000;
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

	int players[MAXAGENTS][1];
};

class AIManager {
public:
	AIManager(){};
	AIManager(int xNum, int yNum, int zNum, float height, float width, float depth);
	~AIManager(){};

	void update(Player* players[], vector<Agent*> allAgents, float msec);
	bool CheckBounding(PhysicsNode& n, float aggroRange,Vector3 pos, Vector3 halfDim);
	void Broadphase(Player* players[], vector<Agent*> allAgents, float msec);
	void Broadphase2(float msec);
	void addAgent(PhysicsNode* a);
	void addPlayer(PhysicsNode* p);

	void init();

protected:
	vector<AIWorldPartition*> allPartitions;
	Vector3 halfDim;

	Agents myAgents;
	Players myPlayers;

	int agentCount;
	int playerCount;

	PhysicsNode* playerNodes[Players::MAXPLAYERS];
	PhysicsNode* agentNodes[Agents::MAXAGENTS];

	vector<int> myAgentsPlayers[Agents::MAXAGENTS];

	Ability agentAbilities[5];

	Players* dev_players;
	Agents* dev_agents;
};

#endif