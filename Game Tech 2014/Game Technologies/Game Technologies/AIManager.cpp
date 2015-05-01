#include "AIManager.h"

vector<AIWorldPartition*> createWorldPartitions(int xNum, int yNum, int zNum, float height, float width, float depth)
{
	//get position locations
	float xDiff = width / (xNum+1);
	float yDiff = height / (yNum+1);
	float zDiff = depth / (zNum+1);

	float xHalf = width / 2;
	float yHalf = height / 2;
	float zHalf = depth / 2;

	vector<AIWorldPartition*> partitions;

	for (int i = 1; i <= xNum; ++i)
	{
		for (int j = 1; j <= yNum; ++j)
		{
			for (int k = 1; k <= zNum; ++k)
			{
				AIWorldPartition* world = new AIWorldPartition();
				world->pos = Vector3((xDiff * i) - xHalf, (yDiff * j) - yHalf, (zDiff * k) - zHalf);
				partitions.push_back(world);
			}
		}
	}

	return partitions;
}

void AIManager::Broadphase(Player* players[], vector<Agent*> allAgents, float msec)
{
	//loop for all world partitions
	for (int i = 0; i < allPartitions.size(); i++) {
		allPartitions[i]->myAgents.clear();
		allPartitions[i]->myPlayers.clear();

		//do the players
		for (int j = 0; j < Player::MAX_PLAYERS; j++) {
			Player* p = players[j];
			if (p != NULL && CheckBounding(*p->physicsNode, 0, allPartitions[i]->pos, halfDim))
			{
				allPartitions[i]->myPlayers.push_back(players[j]);
			}
		}

		//add the agents and update the agents
		for (int j = 0; j < allAgents.size(); j++) {
			Agent* a = allAgents[j];
			if (CheckBounding(*a->physicsNode, Agent::MAXAGGRORANGE, allPartitions[i]->pos, halfDim))
			{
				allPartitions[i]->myAgents.push_back(allAgents[j]);
				allPartitions[i]->myPlayers.resize(Player::MAX_PLAYERS);
				allAgents[j]->Update(&allPartitions[i]->myPlayers[0], msec);
			}
		}
	}
}

AIManager::AIManager(int xNum, int yNum, int zNum, float height, float width, float depth)
{
	halfDim = Vector3(width / (xNum * 2), height / (yNum * 2), depth / (zNum * 2));

	allPartitions = createWorldPartitions(xNum, yNum, zNum, height, width, depth);

}

void AIManager::update(Player* players[], vector<Agent*> allAgents, float msec)
{
	Broadphase(players, allAgents, msec);
}

bool AIManager::CheckBounding(PhysicsNode& n, float aggroRange,Vector3 pos, Vector3 halfDim)
{
	CollisionSphere& sphere = *(CollisionSphere*)n.GetCollisionVolume();

	float dist = abs(pos.x - n.GetPosition().x);
	float sum = halfDim.x + sphere.GetRadius();

	if(dist <= sum) {
		dist = abs(pos.y - n.GetPosition().y);
		sum = halfDim.y + sphere.GetRadius();

		if(dist <= sum) {
			dist = abs(pos.z - n.GetPosition().z);
			sum = halfDim.z + sphere.GetRadius();

			if(dist <= sum) {
				//if there is collision data storage
				return true;
			}
		}
	}
	return false;
}