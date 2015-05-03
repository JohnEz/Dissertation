#include "Agent.h"
#include "Player.h"
#include "PhysicsNode.h"
#include <vector>

struct AIWorldPartition{
	Vector3 pos;
	vector<Agent*> myAgents;
	vector<Player*> myPlayers;
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

	static const int MAXPLAYERS = 5;

	int level[MAXPLAYERS];
	int hp[MAXPLAYERS];
	int maxHP[MAXPLAYERS];
	bool isDead[MAXPLAYERS];
	
	float x[MAXPLAYERS];
	float y[MAXPLAYERS];
	float z[MAXPLAYERS];
	
};

struct Agents {
	static const int MAXAGENTS = 1000;
	static const int AGGRORANGE = 1000;

	State state[MAXAGENTS];
	int targetLocation[MAXAGENTS];
	Vector3 patrolLocation[MAXAGENTS][3];
	int targetPlayer[MAXAGENTS];
	Ability* myAbilities[MAXAGENTS][3];
	int level[MAXAGENTS];

	float x[MAXAGENTS];
	float y[MAXAGENTS];
	float z[MAXAGENTS];
};

class AIManager {
public:
	AIManager(){};
	AIManager(int xNum, int yNum, int zNum, float height, float width, float depth);
	~AIManager(){};

	void update(Player* players[], vector<Agent*> allAgents, float msec);
	bool CheckBounding(PhysicsNode& n, float aggroRange,Vector3 pos, Vector3 halfDim);
	void Broadphase(Player* players[], vector<Agent*> allAgents, float msec);
	void addAgent(PhysicsNode* a);
	void addPlayer(PhysicsNode* p);

protected:
	vector<AIWorldPartition*> allPartitions;
	Vector3 halfDim;

	Agents myAgents;
	Players myPlayers;

	int agentCount;
	int playerCount;

	PhysicsNode* playerNodes[Players::MAXPLAYERS];
	PhysicsNode* agentNodes[Agents::MAXAGENTS];
};