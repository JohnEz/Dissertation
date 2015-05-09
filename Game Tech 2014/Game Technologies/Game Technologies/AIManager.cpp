#include "AIManager.h"
#include "kernal.cuh"
#include "Renderer.h"
#include "PhysicsSystem.h"

void (*states[MAX_STATES]) (int* a, CopyOnce* coreData, CopyEachFrame* updateData, float msec);

AIManager* AIManager::aiInst = 0;


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

	partitions.halfDim = halfDim;

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

void Patrol(int* a, CopyOnce* coreData, CopyEachFrame* updateData, float msec)
{
	float MAXSPEED = 0.5F;

	//at target
	float disX = coreData->myAgents.patrolLocation[*a][coreData->myAgents.targetLocation[*a]].x - coreData->myAgents.x[*a];
	float disZ = coreData->myAgents.patrolLocation[*a][coreData->myAgents.targetLocation[*a]].z - coreData->myAgents.z[*a];
	float absX = abs(disX);
	float absZ = abs(disZ);

	//check its close enough to the point
	if (absX < 0.1f && absZ < 0.1f)
	{
		//get new target
		coreData->myAgents.targetLocation[*a]++;
		coreData->myAgents.targetLocation[*a] = coreData->myAgents.targetLocation[*a] % 2; //need to fix this
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
		coreData->myAgents.x[*a] += moveX * sgn<float>(disX);
		coreData->myAgents.z[*a] += moveZ * sgn<float>(disZ);
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
	while (i < 8 && updateData->agentPartitions[((*a)*8) + i] != -1)
	{
		int j = 0;
		int part = updateData->agentPartitions[(*a*8) + i];
		int partPlayer = (part*coreData->myPlayers.MAXPLAYERS);
		while (j < coreData->myPlayers.MAXPLAYERS && updateData->partitionPlayers[partPlayer+j] != -1)
		{
			//the player
			short p = updateData->partitionPlayers[partPlayer+j];

			//calculate distance to player
			Vector3 diff = Vector3(coreData->myPlayers.x[p] - coreData->myAgents.x[*a], coreData->myPlayers.y[p] - coreData->myAgents.y[*a], coreData->myPlayers.z[p] - coreData->myAgents.z[*a]);

			float dist = sqrtf(Vector3::Dot(diff, diff));

			//if player close transition state to stare at player
			float aggroRange = min(coreData->myAgents.AGGRORANGE, coreData->myAgents.AGGRORANGE * ((float)coreData->myAgents.level[*a] / (float)coreData->myPlayers.level[p]));

			if (dist < aggroRange && !updateData->playerIsDead[p])
			{
				coreData->myAgents.state[*a] = STARE_AT_PLAYER; //change state
				coreData->myAgents.patrolLocation[*a][2] = Vector3(coreData->myAgents.x[*a], coreData->myAgents.y[*a], coreData->myAgents.z[*a]); //set position it left patrol
				coreData->myAgents.targetPlayer[*a] = p; // playing that is being stared at
				i = coreData->myPlayers.MAXPLAYERS; // exit the loop
			}

			++j;
		}

		++i;
	}

}

void stareAtPlayer(int* a, CopyOnce* coreData, CopyEachFrame* updateData, float msec)
{
	int p = coreData->myAgents.targetPlayer[*a]; // target player

	//calculate distance to player
	Vector3 playerPos = Vector3(coreData->myPlayers.x[p], coreData->myPlayers.y[p], coreData->myPlayers.z[p]);
	Vector3 diff = playerPos - Vector3(coreData->myAgents.x[*a], coreData->myAgents.y[*a], coreData->myAgents.z[*a]);
	float dist = sqrtf(Vector3::Dot(diff, diff));
	float aggroRange = min(coreData->myAgents.AGGRORANGE, coreData->myAgents.AGGRORANGE * ((float)coreData->myAgents.level[*a] / (float)coreData->myPlayers.level[p]));
	float pullRange = (aggroRange * 0.75f) * ((float)coreData->myAgents.level[*a] / (float)coreData->myPlayers.level[p]);


	if (dist < pullRange && !updateData->playerIsDead[p]) // if the player is in pull range
	{
		coreData->myAgents.state[*a] = CHASE_PLAYER;
	}
	else
	{
		// if the player isnt in pull range check if there are any players closer
		bool playerClose = false;
		int i = 0;

		while (i < 8 && updateData->agentPartitions[((*a)*8) + i] != -1)
		{
			int j = 0;
			int part = updateData->agentPartitions[(*a*8) + i];
			int partPlayer = (part*coreData->myPlayers.MAXPLAYERS);
			while (j < coreData->myPlayers.MAXPLAYERS && updateData->partitionPlayers[partPlayer+j] != -1)
			{
				//the player
				short p2 = updateData->partitionPlayers[partPlayer+j];

				//calculate distance to player
				playerPos = Vector3(coreData->myPlayers.x[p2], coreData->myPlayers.y[p2], coreData->myPlayers.z[p2]);
				Vector3 diffNew = playerPos - Vector3(coreData->myAgents.x[*a], coreData->myAgents.y[*a], coreData->myAgents.z[*a]);
				float distNew = sqrtf(Vector3::Dot(diffNew, diffNew));

				// if the new distance is less switch targte
				if (distNew <= dist  && !updateData->playerIsDead[p2])
				{
					coreData->myAgents.targetPlayer[*a] = p2;
					dist = distNew;
					float aggroRangeNew = min(coreData->myAgents.AGGRORANGE, coreData->myAgents.AGGRORANGE * (coreData->myAgents.level[*a] / coreData->myPlayers.level[p2]));

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
			coreData->myAgents.state[*a] = PATROL;
			coreData->myAgents.targetPlayer[*a] = -1;
		}
	}
}

void chasePlayer(int* a, CopyOnce* coreData, CopyEachFrame* updateData, float msec)
{
	float LEASHRANGE = 3200.0f;
	float ATTACKRANGE = 75.0f;
	float MAXSPEED = 0.5F;

	int p = coreData->myAgents.targetPlayer[*a];

	//calculate distance to leash spot
	float diffX = coreData->myAgents.patrolLocation[*a][2].x - coreData->myAgents.x[*a];
	float diffY = coreData->myAgents.patrolLocation[*a][2].y - coreData->myAgents.y[*a];
	float diffZ = coreData->myAgents.patrolLocation[*a][2].z - coreData->myAgents.z[*a];

	Vector3 leashDiff = Vector3(diffX, diffY, diffZ);
	float leashDist = sqrtf(Vector3::Dot(leashDiff, leashDiff));

	// if its too far away or if the player died leash back
	if (leashDist > LEASHRANGE || updateData->playerIsDead[p])
	{
		coreData->myAgents.state[*a] = LEASH;
		coreData->myAgents.targetPlayer[*a] = -1;
	}
	else
	{
		//calculate distance to player
		Vector3 playerPos = Vector3(coreData->myPlayers.x[p], coreData->myPlayers.y[p], coreData->myPlayers.z[p]);
		Vector3 diff = playerPos - Vector3(coreData->myAgents.x[*a], coreData->myAgents.y[*a], coreData->myAgents.z[*a]);
		float dist = sqrtf(Vector3::Dot(diff, diff));

		//if close to player switch state to useability
		if (dist < ATTACKRANGE)
		{
			coreData->myAgents.state[*a] = USE_ABILITY;
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
		coreData->myAgents.x[*a] += moveX * sgn<float>(diff.x);
		coreData->myAgents.z[*a] += moveZ * sgn<float>(diff.z);
	}

}

void leashBack(int* a, CopyOnce* coreData, CopyEachFrame* updateData, float msec)
{
	float MAXSPEED = 0.5F;

	//calculate distance to leash spot
	float diffX = coreData->myAgents.patrolLocation[*a][2].x - coreData->myAgents.x[*a];
	float diffZ = coreData->myAgents.patrolLocation[*a][2].z - coreData->myAgents.z[*a];
	float absX = abs(diffX);
	float absZ = abs(diffZ);

	//check its close enough to the point
	if (absX < 0.1f && absZ < 0.1f)
	{
		//change back to patrol
		coreData->myAgents.state[*a] = PATROL;
	}
	else
	{
		//move to target
		float dis = absX + absZ;
		float moveX = ((absX / dis) * MAXSPEED) * msec;
		float moveZ = ((absZ / dis) * MAXSPEED) * msec;

		moveX = min(moveX, absX);
		moveZ = min(moveZ, absZ);

		coreData->myAgents.x[*a] += moveX * sgn<float>(diffX);
		coreData->myAgents.z[*a] += moveZ * sgn<float>(diffZ);
	}
}

void useAbility(int* a, CopyOnce* coreData, CopyEachFrame* updateData, float msec)
{

	float ATTACKRANGE = 75.0f;
	int p = coreData->myAgents.targetPlayer[*a];

	if (updateData->playerIsDead[p]) // if the player is dead
	{
		coreData->myAgents.state[*a] = LEASH;	//leash back
		coreData->myAgents.targetPlayer[*a] = -1; // set the target player to null
	}
	else
	{

		//TODO ADD ABILITIES BACK
		//look through abilities via priority until one is found not on cooldown
		int i = 0;
		while (i < coreData->myAgents.MAXABILITIES && coreData->myAgents.myAbilities[*a][i].cooldown > 0.001f) {
			i++;
		}

		//cast ability
		if (i < coreData->myAgents.MAXABILITIES && coreData->myAgents.myAbilities[*a][i].cooldown < 0.001f)
		{
			coreData->myAgents.myAbilities[*a][i].cooldown = coreData->myAgents.myAbilities[*a][i].maxCooldown;
			coreData->myPlayers.hp[coreData->myAgents.targetPlayer[*a]] -= coreData->myAgents.myAbilities[*a][i].damage;
		}

		//if the player goes out of range, change state to chase
		//calculate distance to player
		//calculate distance to player
		Vector3 playerPos = Vector3(coreData->myPlayers.x[p], coreData->myPlayers.y[p], coreData->myPlayers.z[p]);
		Vector3 diff = playerPos - Vector3(coreData->myAgents.x[*a], coreData->myAgents.y[*a], coreData->myAgents.z[*a]);
		float dist = sqrtf(Vector3::Dot(diff, diff));

		//if player close transition state to stare at player
		if (dist > (ATTACKRANGE))
		{
			coreData->myAgents.state[*a] = CHASE_PLAYER;
		}
	}
}

void reduceCooldowns(int* a, CopyOnce* coreData, float msec)
{
	for (int i = 0 ; i < Agents::MAXABILITIES; ++i)
	{
		if (coreData->myAgents.myAbilities[*a][i].cooldown > 0)
		{
			coreData->myAgents.myAbilities[*a][i].cooldown -= msec;
		}
	}
}

AIManager* AIManager::GetInstance()
{
	if (aiInst == 0)
	{
		aiInst = new AIManager();
	}

	return aiInst;
}

AIManager::~AIManager()
{
	clearCoreData();
}

void AIManager::init(int xNum, int yNum, int zNum, float height, float width, float depth)
{
	Vector3 halfDim = Vector3(width / (xNum * 2), height / (yNum * 2), depth / (zNum * 2));

	coreData.myPartitions = createWorldPartitions(xNum, yNum, zNum, height, width, depth, halfDim);

	agentCount = 0;
	playerCount = 0;

	for (int i = 0; i < coreData.myAgents.MAXAGENTS; ++i)
	{
		agentNodes[i] = NULL;
	}

	broadphaseCounter = 0;

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

	updateData.agentPartitions = new short[Agents::MAXAGENTS*8];
	updateData.partitionPlayers = new short[AIWorldPartition::MAXPARTITIONS*Players::MAXPLAYERS];
	memset(updateData.agentPartitions, -1, (Agents::MAXAGENTS*8) * sizeof(short));
	memset(updateData.partitionPlayers, -1, (AIWorldPartition::MAXPARTITIONS*Players::MAXPLAYERS) * sizeof(short));
}

void AIManager::Broadphase(float msec)
{
	//loop for all world partitions
	for (int i = 0; i < AIWorldPartition::MAXPARTITIONS; ++i) {
		if (coreData.myPartitions.pos[i] != Vector3(0,0,0))
		{
			updateData.playerCount[i] = 0;

			//do the players
			for (int j = 0; j < playerCount; j++) {
				if(!updateData.playerIsDead[j] && coreData.myPlayers.maxHP[j] > 0)
				{

					Vector3 playerPos = Vector3(coreData.myPlayers.x[j], coreData.myPlayers.y[j], coreData.myPlayers.z[j]);

					if (CheckBounding(playerPos, 0, coreData.myPartitions.pos[i], coreData.myPartitions.halfDim))
					{
						updateData.partitionPlayers[(i*Players::MAXPLAYERS) + updateData.playerCount[i]] = j;
						++updateData.playerCount[i];
					}
				}
			}
		}

	}

	int p = 0;

	//add the agents and update the agents
	for (int i = 0; i < agentCount; ++i) {
		p = 0;

		for (int j = 0; j < AIWorldPartition::MAXPARTITIONS; ++j) {
			//check if the agent is in this partition
			if(coreData.myPartitions.pos[j] != Vector3(0,0,0)){

				Vector3 agentPos = Vector3(coreData.myAgents.x[i], coreData.myAgents.y[i], coreData.myAgents.z[i]);

				if (CheckBounding(agentPos, Agents::AGGRORANGE, coreData.myPartitions.pos[j], coreData.myPartitions.halfDim))
				{
					updateData.agentPartitions[(i*8) + p] = j;
					++p;
				}
			}
		}

	}
}

void AIManager::setupCuda()
{
	d_coreData = 0;

	cudaCopyCore(&coreData, &updateData, agentCount, partitionCount, 0);
}

void AIManager::dismantleCuda()
{
	clearCoreData();
}

void AIManager::update(float msec)
{
	//set the node positions after updates
	for (int i = 0; i < agentCount; ++i)
	{
		//run the state functions
		//states[coreData.myAgents.state[i]](&i, &coreData, &updateData, msec);
		reduceCooldowns(&i, &coreData, msec);

		if (agentNodes[i] != NULL)
		{
			agentNodes[i]->target->SetTransform(Matrix4::Translation(Vector3(coreData.myAgents.x[i], coreData.myAgents.y[i], coreData.myAgents.z[i])));
		}
	}

	//update players
	for (int i = 0; i < playerCount; ++i)
	{
		// if the player isnt dead
		if (!updateData.playerIsDead[i])
		{
			//set the nodes position to the players position
			//playerNodes[i]->SetPosition(Vector3(coreData.myPlayers.x[i], coreData.myPlayers.y[i], coreData.myPlayers.z[i]));
			playerNodes[i]->target->SetTransform(Matrix4::Translation(Vector3(coreData.myPlayers.x[i], coreData.myPlayers.y[i], coreData.myPlayers.z[i])));

			// if the player's hp is 0
			if (coreData.myPlayers.hp[i] < 1)
			{
				// the player must be dead
				updateData.playerIsDead[i] = true;
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

	if (broadphaseCounter == 0)
	{
		Broadphase(msec);
		broadphaseCounter = 10;
	}

	--broadphaseCounter;

	//cudaUpdateAgents(&coreData, &updateData, agentCount, partitionCount, msec);

	cudaRunKernal(&coreData, &updateData, agentCount, partitionCount, msec);
	copyDataFromGPU(&coreData, &updateData, agentCount, partitionCount, msec);

	/*if (Window::GetKeyboard()->KeyDown(KEYBOARD_UP))
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
	}*/
}

bool AIManager::CheckBounding(const Vector3& n, float aggroRange,Vector3 pos, Vector3 halfDim)
{
	float dist = abs(pos.x - n.x);
	float sum = halfDim.x + aggroRange;



	if(dist <= sum) {
		dist = abs(pos.y - n.y);
		sum = halfDim.y + aggroRange;

		if(dist <= sum) {
			dist = abs(pos.z - n.z);
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
	coreData.myAgents.state[agentCount] = PATROL; // starting state

	coreData.myAgents.targetLocation[agentCount] = 0;

	coreData.myAgents.patrolLocation[agentCount][0] = GenerateTargetLocation(a->GetPosition());	// start patrol
	//myAgents.patrolLocation[agentCount][0] = Vector3(0, 0, 0);
	coreData.myAgents.patrolLocation[agentCount][1] = GenerateTargetLocation(a->GetPosition());	// end patrol
	coreData.myAgents.patrolLocation[agentCount][2] = Vector3(0, 0, 0);							// store location

	coreData.myAgents.targetPlayer[agentCount] = -1; // no target player

	coreData.myAgents.myAbilities[agentCount][0] = agentAbilities[0];
	coreData.myAgents.myAbilities[agentCount][1] = agentAbilities[1];
	coreData.myAgents.myAbilities[agentCount][2] = agentAbilities[4];

	coreData.myAgents.level[agentCount] = 100; //(rand() % 100) + 1; // randomly generate level

	coreData.myAgents.x[agentCount] = a->GetPosition().x; // store the x in an array
	coreData.myAgents.y[agentCount] = a->GetPosition().y; // store the y in an array
	coreData.myAgents.z[agentCount] = a->GetPosition().z; // store the z in an array

	agentNodes[agentCount] = a; // store the physic nodes for updating after cuda

	if (agentCount < Agents::MAXAGENTS - 1) // TODO probably should move this to the top
	{
		agentCount++;
	}
	Performance::GetInstance()->setScore(agentCount);


}

void AIManager::addPlayer(PhysicsNode* p)
{
	coreData.myPlayers.level[playerCount] = 100; //(rand() % 100) + 1; // randomly generate level

	coreData.myPlayers.hp[playerCount] = 2000; //set hp

	coreData.myPlayers.maxHP[playerCount] = 2000; //set the max hp

	updateData.playerIsDead[playerCount] = false; // make the player alive

	coreData.myPlayers.x[playerCount] = p->GetPosition().x; // store the x in an array
	coreData.myPlayers.y[playerCount] = p->GetPosition().y; // store the y in an array
	coreData.myPlayers.z[playerCount] = p->GetPosition().z; // store the z in an array

	playerNodes[playerCount] = p;

	if (playerCount < Players::MAXPLAYERS - 1) // TODO probably should move this to the top
	{
		playerCount++;
	}

	Performance::GetInstance()->setCollisions(playerCount);
}