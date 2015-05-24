#include "PhysicsNode.h"
#include <vector>

#ifndef AIMANAGER
#define AIMANAGER

//#define NO_AI
//#define BASICCPU
//#define BASICGPU
//#define LESS_DATA_GPU
//#define GPU_OLD_BROAD
//#define GPU_NEW_BROAD
//#define GPU_BROAD_AGENTS
#define SPLIT_GPU
//#define SPLIT_GPU_BROAD
//#define SORT_SPLIT

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
	
	float x[MAXPLAYERS];
	float y[MAXPLAYERS];
	float z[MAXPLAYERS];
	
};

struct AIWorldPartition {
	static const int MAXPARTITIONS = 100;
	Vector3 halfDim;
	Vector3 pos[MAXPARTITIONS];
};

struct PatrolLocations {
	Vector3 loc[3];
};

struct AgentAbilities {
	static const int MAXABILITIES = 3;
	Ability abil[MAXABILITIES];
};

struct Agents {
	static const int MAXAGENTS = 1024 * 1000;
	static const int AGGRORANGE = 500;

	State state[MAXAGENTS];
	int targetLocation[MAXAGENTS];
	PatrolLocations patrolLocation[MAXAGENTS];
	int targetPlayer[MAXAGENTS];
	AgentAbilities myAbilities[MAXAGENTS];
	int level[MAXAGENTS];

	float x[MAXAGENTS];
	float y[MAXAGENTS];
	float z[MAXAGENTS];

	int partCount[MAXAGENTS];
	float waitedTime[MAXAGENTS];

	int stateCount[MAX_STATES];
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
	bool playerIsDead[Players::MAXPLAYERS];

	//agents
	short* agentPartitions;

	//partitions
	short* partitionPlayers; //players in the partition
	int playerCount[AIWorldPartition::MAXPARTITIONS];	 //number of players in the partition
};



class AIManager {
public:
	~AIManager();

	static AIManager* GetInstance();

	void init(int xNum, int yNum, int zNum, float height, float width, float depth);

	void update(float msec);
	bool CheckBounding(const Vector3& n, float aggroRange,Vector3 pos, Vector3 halfDim);
	void Broadphase();
	void addAgent(PhysicsNode* a);
	void addPlayer(PhysicsNode* p);

	void setupCuda();
	void dismantleCuda();

	CopyOnce* d_coreData;
	CopyEachFrame* d_updateData;
	

protected:

	static AIManager* aiInst;

	vector<AIWorldPartition> allPartitions;

	CopyOnce coreData;
	CopyEachFrame updateData;

	int agentCount;
	int playerCount;
	int partitionCount;

	PhysicsNode* playerNodes[Players::MAXPLAYERS];
	PhysicsNode* agentNodes[Agents::MAXAGENTS];

	vector<int> myAgentsPlayers[Agents::MAXAGENTS];

	Ability agentAbilities[5];

	unsigned int broadphaseCounter;

	float timeSinceBroad;
};

#endif