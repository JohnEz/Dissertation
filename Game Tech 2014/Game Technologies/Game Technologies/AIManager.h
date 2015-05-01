#include "Agent.h"
#include "Player.h"
#include <vector>

struct AIWorldPartition{
	Vector3 pos;
	vector<Agent*> myAgents;
	vector<Player*> myPlayers;
};

class AIManager {
public:
	AIManager(){};
	AIManager(int xNum, int yNum, int zNum, float height, float width, float depth);
	~AIManager(){};

	void update(Player* players[], vector<Agent*> allAgents, float msec);
	bool CheckBounding(PhysicsNode& n, float aggroRange,Vector3 pos, Vector3 halfDim);
	void Broadphase(Player* players[], vector<Agent*> allAgents, float msec);
protected:
	vector<AIWorldPartition*> allPartitions;
	Vector3 halfDim;
};