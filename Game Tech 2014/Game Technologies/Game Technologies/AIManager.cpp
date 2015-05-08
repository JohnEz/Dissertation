#include "AIManager.h"
#include "kernal.cuh"
#include "Renderer.h"
#include "PhysicsSystem.h"

void (*states[MAX_STATES]) (int* a, Players* players, Agents* agents, AIWorldPartition* partitions, float msec);

AIManager* AIManager::aiInst = 0;


/*vector<AIWorldPartition> createWorldPartitions(int xNum, int yNum, int zNum, float height, float width, float depth, const Vector3 halfDim)
{
//get position locations
float xDiff = width / (xNum);
float yDiff = height / (yNum);
float zDiff = depth / (zNum);

float xHalf = width / 2;
float yHalf = height / 2;
float zHalf = depth / 2;

vector<AIWorldPartition> partitions;

for (int i = 1; i <= xNum; ++i)
{
for (int j = 1; j <= yNum; ++j)
{
for (int k = 1; k <= zNum; ++k)
{
AIWorldPartition world = AIWorldPartition();
world.pos = Vector3((xDiff * i) - xHalf - halfDim.x, (yDiff * j) - yHalf - halfDim.x, (zDiff * k) - zHalf - halfDim.x);
partitions.push_back(world);
}
}
}

return partitions;
}*/

AIWorldPartition createWorldPartitions(int xNum, int yNum, int zNum, float height, float width, float depth, const Vector3 halfDim)
{
	//get position locations
	float xDiff = width / (xNum);
	float yDiff = height / (yNum);
	float zDiff = depth / (zNum);

	float xHalf = width / 2;
	float yHalf = height / 2;
	float zHalf = depth / 2;

	AIWorldPartition partitions;

	int count = 0;

	for (int i = 1; i <= xNum; ++i)
	{
		for (int j = 1; j <= yNum; ++j)
		{
			for (int k = 1; k <= zNum; ++k)
			{
				partitions.pos[count] = Vector3((xDiff * i) - xHalf - halfDim.x, (yDiff * j) - yHalf - halfDim.y, (zDiff * k) - zHalf - halfDim.z);
				++count;
			}
		}
	}

	return partitions;
}

template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

Matrix4		BuildTransform(const PhysicsNode &node) {
	Matrix4 m = node.GetOrientation().ToMatrix();

	m.SetPositionVector(node.GetPosition());

	return m;
}

Vector3 GenerateTargetLocation(const Vector3& position)
{
	Vector3 target = position;

	target.x += (rand() % 4000) - 2000;
	target.z += (rand() % 4000) - 2000;

	return target;
}

void Patrol(int* a, Players* players, Agents* agents, AIWorldPartition* partitions, float msec)
{
	float MAXSPEED = 0.5F;

	//at target
	float disX = agents->patrolLocation[*a][agents->targetLocation[*a]].x - agents->x[*a];
	float disZ = agents->patrolLocation[*a][agents->targetLocation[*a]].z - agents->z[*a];
	float absX = abs(disX);
	float absZ = abs(disZ);

	//check its close enough to the point
	if (absX < 0.1f && absZ < 0.1f)
	{
		//get new target
		agents->targetLocation[*a]++;
		agents->targetLocation[*a] = agents->targetLocation[*a] % 2; //need to fix this
	}
	else
	{
		//move to target
		float dis = absX + absZ;
		float moveX = ((absX / dis) * MAXSPEED) * msec;
		float moveZ = ((absZ / dis) * MAXSPEED) * msec;

		//find how much it needs to move
		moveX = min(moveX, absX);
		moveZ = min(moveZ, absZ);

		//set new position
		agents->x[*a] += moveX * sgn<float>(disX);
		agents->z[*a] += moveZ * sgn<float>(disZ);
	}

	//state transition

	/*int i = 0;
	// loop through all the players
	while (i < players->MAXPLAYERS && agents->players[*a][i] > -1)
	{

	//the player
	int p = agents->players[*a][i];
	//calculate distance to player
	Vector3 diff = Vector3(players->x[p] - agents->x[*a], players->y[p] - agents->y[*a],  players->z[p] - agents->z[*a]);

	float dist = sqrtf(Vector3::Dot(diff, diff));

	//if player close transition state to stare at player
	float aggroRange = min(agents->AGGRORANGE, agents->AGGRORANGE * ((float)agents->level[*a] / (float)players->level[p]));

	if (dist < aggroRange && !players->isDead[p])
	{
	agents->state[*a] = STARE_AT_PLAYER; //change state
	agents->patrolLocation[*a][2] = Vector3(agents->x[*a], agents->y[*a], agents->z[*a]); //set position it left patrol
	agents->targetPlayer[*a] = p; // playing that is being stared at
	i = players->MAXPLAYERS; // exit the loop
	}
	i++;
	}*/

	int i = 0;
	while (i < 8 && agents->partitions[((*a)*8) + i] != -1)
	{
		int j = 0;
		int part = agents->partitions[(*a*8) + i];
		int partPlayer = (part*players->MAXPLAYERS);
		while (j < players->MAXPLAYERS && partitions->myPlayers[partPlayer+j] != -1)
		{
			//the player
			short p = partitions->myPlayers[partPlayer+j];

			//calculate distance to player
			Vector3 diff = Vector3(players->x[p] - agents->x[*a], players->y[p] - agents->y[*a],  players->z[p] - agents->z[*a]);

			float dist = sqrtf(Vector3::Dot(diff, diff));

			//if player close transition state to stare at player
			float aggroRange = min(agents->AGGRORANGE, agents->AGGRORANGE * ((float)agents->level[*a] / (float)players->level[p]));

			if (dist < aggroRange && !players->isDead[p])
			{
				agents->state[*a] = STARE_AT_PLAYER; //change state
				agents->patrolLocation[*a][2] = Vector3(agents->x[*a], agents->y[*a], agents->z[*a]); //set position it left patrol
				agents->targetPlayer[*a] = p; // playing that is being stared at
				i = players->MAXPLAYERS; // exit the loop
			}

			++j;
		}

		++i;
	}

}

void stareAtPlayer(int* a, Players* players, Agents* agents, AIWorldPartition* partitions, float msec)
{
	int p = agents->targetPlayer[*a]; // target player

	//calculate distance to player
	Vector3 playerPos = Vector3(players->x[p], players->y[p], players->z[p]);
	Vector3 diff = playerPos - Vector3(agents->x[*a], agents->y[*a], agents->z[*a]);
	float dist = sqrtf(Vector3::Dot(diff, diff));
	float aggroRange = min(agents->AGGRORANGE, agents->AGGRORANGE * ((float)agents->level[*a] / (float)players->level[p]));
	float pullRange = (aggroRange * 0.75f) * ((float)agents->level[*a] / (float)players->level[p]);


	if (dist < pullRange && !players->isDead[p]) // if the player is in pull range
	{
		agents->state[*a] = CHASE_PLAYER;
	}
	else
	{
		// if the player isnt in pull range check if there are any players closer
		bool playerClose = false;
		int i = 0;

		while (i < 8 && agents->partitions[((*a)*8) + i] != -1)
		{
			int j = 0;
			int part = agents->partitions[(*a*8) + i];
			int partPlayer = (part*players->MAXPLAYERS);
			while (j < players->MAXPLAYERS && partitions->myPlayers[partPlayer+j] != -1)
			{
				//the player
				short p2 = partitions->myPlayers[partPlayer+j];

				//calculate distance to player
				playerPos = Vector3(players->x[p2], players->y[p2], players->z[p2]);
				Vector3 diffNew = playerPos - Vector3(agents->x[*a], agents->y[*a], agents->z[*a]);
				float distNew = sqrtf(Vector3::Dot(diffNew, diffNew));

				// if the new distance is less switch targte
				if (distNew <= dist  && !players->isDead[p2])
				{
					agents->targetPlayer[*a] = p2;
					dist = distNew;
					float aggroRangeNew = min(agents->AGGRORANGE, agents->AGGRORANGE * (agents->level[*a] / players->level[p2]));

					if (dist < aggroRangeNew)
					{
						playerClose = true;
					}
				}
				++j;
			}
			++i;
		}

		// if there are no close players at all
		if (!playerClose)
		{
			agents->state[*a] = PATROL;
			agents->targetPlayer[*a] = -1;
		}
	}
}

void chasePlayer(int* a, Players* players, Agents* agents, AIWorldPartition* partitions, float msec)
{
	float LEASHRANGE = 3200.0f;
	float ATTACKRANGE = 75.0f;
	float MAXSPEED = 0.5F;

	int p = agents->targetPlayer[*a];

	//calculate distance to leash spot
	float diffX = agents->patrolLocation[*a][2].x - agents->x[*a];
	float diffY = agents->patrolLocation[*a][2].y - agents->y[*a];
	float diffZ = agents->patrolLocation[*a][2].z - agents->z[*a];

	Vector3 leashDiff = Vector3(diffX, diffY, diffZ);
	float leashDist = sqrtf(Vector3::Dot(leashDiff, leashDiff));

	// if its too far away or if the player died leash back
	if (leashDist > LEASHRANGE || players->isDead[p])
	{
		agents->state[*a] = LEASH;
		agents->targetPlayer[*a] = -1;
	}
	else
	{
		//calculate distance to player
		Vector3 playerPos = Vector3(players->x[p], players->y[p], players->z[p]);
		Vector3 diff = playerPos - Vector3(agents->x[*a], agents->y[*a], agents->z[*a]);
		float dist = sqrtf(Vector3::Dot(diff, diff));

		//if close to player switch state to useability
		if (dist < ATTACKRANGE)
		{
			agents->state[*a] = USE_ABILITY;
		}

		//move towards players location
		float absX = abs(diff.x);
		float absZ = abs(diff.z);

		//move to target
		float dis = absX + absZ;
		float moveX = ((absX / dis) * MAXSPEED) * msec;
		float moveZ = ((absZ / dis) * MAXSPEED) * msec;

		moveX = min(moveX, absX);
		moveZ = min(moveZ, absZ);

		//set new position
		agents->x[*a] += moveX * sgn<float>(diff.x);
		agents->z[*a] += moveZ * sgn<float>(diff.z);
	}

}

void leashBack(int* a, Players* players, Agents* agents, AIWorldPartition* partitions, float msec)
{
	float MAXSPEED = 0.5F;

	//calculate distance to leash spot
	float diffX = agents->patrolLocation[*a][2].x - agents->x[*a];
	float diffZ = agents->patrolLocation[*a][2].z - agents->z[*a];
	float absX = abs(diffX);
	float absZ = abs(diffZ);

	//check its close enough to the point
	if (absX < 0.1f && absZ < 0.1f)
	{
		//change back to patrol
		agents->state[*a] = PATROL;
	}
	else
	{
		//move to target
		float dis = absX + absZ;
		float moveX = ((absX / dis) * MAXSPEED) * msec;
		float moveZ = ((absZ / dis) * MAXSPEED) * msec;

		moveX = min(moveX, absX);
		moveZ = min(moveZ, absZ);

		agents->x[*a] += moveX * sgn<float>(diffX);
		agents->z[*a] += moveZ * sgn<float>(diffZ);
	}
}

void useAbility(int* a, Players* players, Agents* agents, AIWorldPartition* partitions, float msec)
{

	float ATTACKRANGE = 75.0f;
	int p = agents->targetPlayer[*a];

	if (players->isDead[p]) // if the player is dead
	{
		agents->state[*a] = LEASH;	//leash back
		agents->targetPlayer[*a] = -1; // set the target player to null
	}
	else
	{

		//TODO ADD ABILITIES BACK
		//look through abilities via priority until one is found not on cooldown
		int i = 0;
		while (i < agents->MAXABILITIES && agents->myAbilities[*a][i].cooldown > 0.001f) {
			i++;
		}

		//cast ability
		if (i < agents->MAXABILITIES && agents->myAbilities[*a][i].cooldown < 0.001f)
		{
			agents->myAbilities[*a][i].cooldown = agents->myAbilities[*a][i].maxCooldown;
			players->hp[agents->targetPlayer[*a]] -= agents->myAbilities[*a][i].damage;
			//printf("Ability: %d \n", i);
			//printf("Cooldown: %4.2f \n", agents->myAbilities[*a][i].cooldown);
		}

		//if the player goes out of range, change state to chase
		//calculate distance to player
		//calculate distance to player
		Vector3 playerPos = Vector3(players->x[p], players->y[p], players->z[p]);
		Vector3 diff = playerPos - Vector3(agents->x[*a], agents->y[*a], agents->z[*a]);
		float dist = sqrtf(Vector3::Dot(diff, diff));

		//if player close transition state to stare at player
		if (dist > (ATTACKRANGE))
		{
			agents->state[*a] = CHASE_PLAYER;
		}
	}
}

void reduceCooldowns(int* a, Agents* agents, float msec)
{
	for (int i = 0 ; i < agents->MAXABILITIES; ++i)
	{
		if (agents->myAbilities[*a][i].cooldown > 0)
		{
			agents->myAbilities[*a][i].cooldown -= msec;
		}
	}
}


/*AIManager::AIManager(int xNum, int yNum, int zNum, float height, float width, float depth)
{
halfDim = Vector3(width / (xNum * 2), height / (yNum * 2), depth / (zNum * 2));

allPartitions = createWorldPartitions(xNum, yNum, zNum, height, width, depth, halfDim);

agentCount = 0;
playerCount = 0;

for (int i = 0; i < myAgents.MAXAGENTS; ++i)
{
agentNodes[i] = NULL;
}

states[PATROLS] = Patrol;
states[STARE_AT_PLAYERS] = stareAtPlayer;
states[CHASE_PLAYERS] = chasePlayer;
states[LEASHS] = leashBack;
states[USE_ABILITYS] = useAbility;

agentAbilities[0] = Ability();
agentAbilities[0].maxCooldown = 20000.0f;
agentAbilities[0].damage = 240;
agentAbilities[0].targetEnemy = true;

agentAbilities[1] = Ability();
agentAbilities[1].maxCooldown = 14000.0f;
agentAbilities[1].damage = 140;
agentAbilities[1].targetEnemy = true;

agentAbilities[4] = Ability();
agentAbilities[4].maxCooldown = 1000.0f;
agentAbilities[4].damage = 9;
agentAbilities[4].targetEnemy = true;

}*/

AIManager* AIManager::GetInstance()
{
	if (aiInst == 0)
	{
		aiInst = new AIManager();
	}

	return aiInst;
}

void AIManager::init(int xNum, int yNum, int zNum, float height, float width, float depth)
{
	halfDim = Vector3(width / (xNum * 2), height / (yNum * 2), depth / (zNum * 2));

	myPartitions;
	myPartitions = AIWorldPartition();
	myPartitions = createWorldPartitions(xNum, yNum, zNum, height, width, depth, halfDim);

	agentCount = 0;
	playerCount = 0;

	for (int i = 0; i < myAgents.MAXAGENTS; ++i)
	{
		agentNodes[i] = NULL;
	}

	states[PATROL] = Patrol;
	states[STARE_AT_PLAYER] = stareAtPlayer;
	states[CHASE_PLAYER] = chasePlayer;
	states[LEASH] = leashBack;
	states[USE_ABILITY] = useAbility;

	agentAbilities[0] = Ability();
	agentAbilities[0].maxCooldown = 20000.0f;
	agentAbilities[0].damage = 240;
	agentAbilities[0].targetEnemy = true;

	agentAbilities[1] = Ability();
	agentAbilities[1].maxCooldown = 14000.0f;
	agentAbilities[1].damage = 140;
	agentAbilities[1].targetEnemy = true;

	agentAbilities[4] = Ability();
	agentAbilities[4].maxCooldown = 1000.0f;
	agentAbilities[4].damage = 9;
	agentAbilities[4].targetEnemy = true;

	myAgents.partitions = new short[myAgents.MAXAGENTS*8];
	myPartitions.myPlayers = new short[myPartitions.MAXPARTITIONS*Players::MAXPLAYERS];
	memset(myAgents.partitions, -1, (myAgents.MAXAGENTS*8) * sizeof(short));
	memset(myPartitions.myPlayers, -1, (myPartitions.MAXPARTITIONS*Players::MAXPLAYERS) * sizeof(short));
}

void AIManager::Broadphase(float msec)
{
	//loop for all world partitions
	for (int i = 0; i < myPartitions.MAXPARTITIONS; ++i) {
		if (myPartitions.pos[i] != Vector3(0,0,0))
		{
			myPartitions.playerCount[i] = 0;

			//do the players
			for (int j = 0; j < playerCount; j++) {

				if (!myPlayers.isDead[j] && myPlayers.maxHP[j] > 0 && CheckBounding(*playerNodes[j], 0, myPartitions.pos[i], halfDim))
				{
					myPartitions.myPlayers[(i*myPlayers.MAXPLAYERS) + myPartitions.playerCount[i]] = j;
					++myPartitions.playerCount[i];
				}
			}
		}

	}

	int p = 0;
	//memset(myAgents.partitions, -1, (myAgents.MAXAGENTS*8) * sizeof(short));
	//add the agents and update the agents
	for (int i = 0; i < agentCount; ++i) {
		p = 0;

		for (int j = 0; j < myPartitions.MAXPARTITIONS; ++j) {
			//check if the agent is in this partition
			if (myPartitions.pos[j] != Vector3(0,0,0) && CheckBounding(*agentNodes[i], myAgents.AGGRORANGE, myPartitions.pos[j], halfDim))
			{
				myAgents.partitions[(i*8) + p] = j;
				++p;
			}
		}

	}
}

void AIManager::update(float msec)
{
	Broadphase(msec);

	Players* d_players;
	Agents* d_agents;

	cudaUpdateAgents(&myPlayers, &myAgents, agentCount, msec, &myPartitions, partitionCount, &halfDim);

	//addDataToGPU(&myPlayers, &myAgents, agentCount, msec, d_players, d_agents);
	//runKernal(&myPlayers, &myAgents, agentCount, msec, d_players, d_agents);
	//clearData(&myPlayers, &myAgents, agentCount, msec, d_players, d_agents);

	//set the node positions after updates
	for (int i = 0; i < agentCount; ++i)
	{
		//run the state functions
		//states[myAgents.state[i]](&i, &myPlayers, &myAgents, &myPartitions, msec);
		reduceCooldowns(&i, &myAgents, msec);

		if (agentNodes[i] != NULL)
		{
			//agentNodes[i]->SetPosition(Vector3(myAgents.x[i], myAgents.y[i], myAgents.z[i]));
			agentNodes[i]->target->SetTransform(Matrix4::Translation(Vector3(myAgents.x[i], myAgents.y[i], myAgents.z[i])));
		}
	}

	//update players
	for (int i = 0; i < playerCount; ++i)
	{
		// if the player isnt dead
		if (!myPlayers.isDead[i])
		{
			//set the nodes position to the players position
			playerNodes[i]->SetPosition(Vector3(myPlayers.x[i], myPlayers.y[i], myPlayers.z[i]));
			playerNodes[i]->target->SetTransform(Matrix4::Translation(Vector3(myPlayers.x[i], myPlayers.y[i], myPlayers.z[i])));

			// if the player's hp is 0
			if (myPlayers.hp[i] < 1)
			{
				// the player must be dead
				myPlayers.isDead[i] = true;
				if (playerNodes[i])
				{
					if (playerNodes[i]->target)
					{
						Renderer::GetRenderer().RemoveNode(playerNodes[i]->target);
					}
					//PhysicsSystem::GetPhysicsSystem().RemoveNode(playerNodes[i]);
				}
			}
		}

	}

	//Broadphase(msec);


	if (Window::GetKeyboard()->KeyDown(KEYBOARD_UP))
	{
		myPlayers.z[0] -= (1 * msec);
	}

	if (Window::GetKeyboard()->KeyDown(KEYBOARD_RIGHT))
	{
		myPlayers.x[0] += (1 * msec);
	}

	if (Window::GetKeyboard()->KeyDown(KEYBOARD_LEFT))
	{
		myPlayers.x[0] -= (1 * msec);
	}

	if (Window::GetKeyboard()->KeyDown(KEYBOARD_DOWN))
	{
		myPlayers.z[0] += (1 * msec);
	}
}

bool AIManager::CheckBounding(PhysicsNode& n, float aggroRange,Vector3 pos, Vector3 halfDim)
{
	Vector3 nPos = n.target->GetTransform().GetPositionVector();

	float dist = abs(pos.x - nPos.x);
	float sum = halfDim.x + aggroRange;



	if(dist <= sum) {
		dist = abs(pos.y - nPos.y);
		sum = halfDim.y + aggroRange;

		if(dist <= sum) {
			dist = abs(pos.z - nPos.z);
			sum = halfDim.z + aggroRange;

			if(dist <= sum) {
				//if there is collision data storage
				return true;
			}
		}
	}
	return false;
}

void AIManager::addAgent(PhysicsNode* a)
{
	myAgents.state[agentCount] = PATROL; // starting state

	myAgents.targetLocation[agentCount] = 0;

	myAgents.patrolLocation[agentCount][0] = GenerateTargetLocation(a->GetPosition());	// start patrol
	//myAgents.patrolLocation[agentCount][0] = Vector3(0, 0, 0);
	myAgents.patrolLocation[agentCount][1] = GenerateTargetLocation(a->GetPosition());	// end patrol
	myAgents.patrolLocation[agentCount][2] = Vector3(0, 0, 0);							// store location

	myAgents.targetPlayer[agentCount] = -1; // no target player

	myAgents.myAbilities[agentCount][0] = agentAbilities[0];
	myAgents.myAbilities[agentCount][1] = agentAbilities[1];
	myAgents.myAbilities[agentCount][2] = agentAbilities[4];

	myAgents.level[agentCount] = 100; //(rand() % 100) + 1; // randomly generate level

	myAgents.x[agentCount] = a->GetPosition().x; // store the x in an array
	myAgents.y[agentCount] = a->GetPosition().y; // store the y in an array
	myAgents.z[agentCount] = a->GetPosition().z; // store the z in an array

	agentNodes[agentCount] = a; // store the physic nodes for updating after cuda

	if (agentCount < Agents::MAXAGENTS - 1) // TODO probably should move this to the top
	{
		agentCount++;
	}
	Performance::GetInstance()->setScore(agentCount);


}

void AIManager::addPlayer(PhysicsNode* p)
{
	myPlayers.level[playerCount] = 100; //(rand() % 100) + 1; // randomly generate level

	myPlayers.hp[playerCount] = 20000; //set hp

	myPlayers.maxHP[playerCount] = 20000; //set the max hp

	myPlayers.isDead[playerCount] = false; // make the player alive

	myPlayers.x[playerCount] = p->GetPosition().x; // store the x in an array
	myPlayers.y[playerCount] = p->GetPosition().y; // store the y in an array
	myPlayers.z[playerCount] = p->GetPosition().z; // store the z in an array

	playerNodes[playerCount] = p;

	if (playerCount < Players::MAXPLAYERS - 1) // TODO probably should move this to the top
	{
		playerCount++;
	}

	Performance::GetInstance()->setCollisions(playerCount);
}