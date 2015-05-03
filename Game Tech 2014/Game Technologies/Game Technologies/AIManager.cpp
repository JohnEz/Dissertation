#include "AIManager.h"

void (*states[MAX_STATESS]) (int* a, Players* players, Agents* agents, float msec);

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

template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}



Vector3 GenerateTargetLocation(const Vector3& position)
{
	Vector3 target = position;

	target.x += (rand() % 4000) - 2000;
	target.z += (rand() % 4000) - 2000;

	return target;
}

void Patrol(int* a, Players* players, Agents* agents, float msec)
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

	int i = 0;
	// loop through all the players
	while (i < Players::MAXPLAYERS && players->maxHP[i] != 0)
	{
		//calculate distance to player
		Vector3 diff = Vector3(players->x[i] - agents->x[*a], players->y[i] - agents->y[*a],  players->z[i] - agents->z[*a]);

		float dist = sqrtf(Vector3::Dot(diff, diff));

		//if player close transition state to stare at player
		float aggroRange = max(agents->AGGRORANGE, agents->AGGRORANGE * (agents->level[*a] / players->level[i]));

		if (dist < aggroRange && !players->isDead[i])
		{
			agents->state[*a] = STARE_AT_PLAYERS; //change state
			agents->patrolLocation[*a][2] = Vector3(agents->x[*a], agents->y[*a], agents->z[*a]); //set position it left patrol
			agents->targetPlayer[*a] = i; // playing that is being stared at
			i = Player::MAX_PLAYERS; // exit the loop
		}
		i++;
	}

}

void stareAtPlayer(int* a, Players* players, Agents* agents, float msec)
{
	int p = agents->targetPlayer[*a]; // target player

	//calculate distance to player
	Vector3 playerPos = Vector3(players->x[p], players->y[p], players->z[p]);
	Vector3 diff = playerPos - Vector3(agents->x[*a], agents->y[*a], agents->z[*a]);
	float dist = sqrtf(Vector3::Dot(diff, diff));
	float aggroRange = max(agents->AGGRORANGE, agents->AGGRORANGE * (agents->level[*a] / players->level[p]));
	float pullRange = (aggroRange * 0.75f) * ((float)agents->level[*a] / (float)players->level[p]);


	if (dist < pullRange && !players->isDead[p]) // if the player is in pull range
	{
		agents->state[*a] = CHASE_PLAYERS;
	}
	else
	{
		// if the player isnt in pull range check if there are any players closer
		bool playerClose = false;
		int i = 0;

		//loop through the players TODO CHECK ISDEAD SHOULD BE HERE
		while (i < players->MAXPLAYERS && players->maxHP[i] != 0 && !players->isDead[i])
		{
			//calculate distance to player
			playerPos = Vector3(players->x[i], players->y[i], players->z[i]);
			Vector3 diffNew = playerPos - Vector3(agents->x[*a], agents->y[*a], agents->z[*a]);
			float distNew = sqrtf(Vector3::Dot(diffNew, diffNew));

			// if the new distance is less switch targte
			if (distNew <= dist)
			{
				agents->targetPlayer[*a] = i;
				dist = distNew;
				float aggroRangeNew = max(agents->AGGRORANGE, agents->AGGRORANGE * (agents->level[*a] / players->level[i]));

				if (dist < aggroRangeNew)
				{
					playerClose = true;
				}
			}
			++i;
		}

		// if there are no close players at all
		if (!playerClose)
		{
			agents->state[*a] = PATROLS;
			agents->targetPlayer[*a] = -1;
		}
	}
}

void chasePlayer(int* a, Players* players, Agents* agents, float msec)
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
		agents->state[*a] = LEASHS;
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
			agents->state[*a] = USE_ABILITYS;
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

void leashBack(int* a, Players* players, Agents* agents, float msec)
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
		agents->state[*a] = PATROLS;
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

void useAbility(int* a, Players* players, Agents* agents, float msec)
{

	float ATTACKRANGE = 75.0f;
	int p = agents->targetPlayer[*a];

	if (players->isDead[p]) // if the player is dead
	{
		agents->state[*a] = LEASHS;	//leash back
		agents->targetPlayer[*a] = -1; // set the target player to null
	}
	else
	{

		//TODO ADD ABILITIES BACK
		//look through abilities via priority until one is found not on cooldown
		/*int i = 0;
		while (i < MAXABILITIES && a.myAbilities[i]->cooldown > 0.001f)
		{
		i++;
		}

		//cast ability
		if (i < MAXABILITIES && a.myAbilities[i]->cooldown < 0.001f)
		{
		a.myAbilities[i]->cooldown = a.myAbilities[i]->maxCooldown;
		a.targetPlayer->hp -= a.myAbilities[i]->damage;
		printf("Ability: %d \n", i);
		printf("Cooldown: %4.2f \n", a.myAbilities[i]->cooldown);
		}*/

		//if the player goes out of range, change state to chase
		//calculate distance to player
		//calculate distance to player
		Vector3 playerPos = Vector3(players->x[p], players->y[p], players->z[p]);
		Vector3 diff = playerPos - Vector3(agents->x[*a], agents->y[*a], agents->z[*a]);
		float dist = sqrtf(Vector3::Dot(diff, diff));

		//if player close transition state to stare at player
		if (dist > (ATTACKRANGE))
		{
			agents->state[*a] = CHASE_PLAYERS;
		}
	}
}



AIManager::AIManager(int xNum, int yNum, int zNum, float height, float width, float depth)
{
	halfDim = Vector3(width / (xNum * 2), height / (yNum * 2), depth / (zNum * 2));

	allPartitions = createWorldPartitions(xNum, yNum, zNum, height, width, depth);

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

void AIManager::update(Player* players[], vector<Agent*> allAgents, float msec)
{
	Broadphase(players, allAgents, msec);



	//set the node positions after updates
	for (int i = 0; i < agentCount; ++i)
	{
		//run the state functions
		states[myAgents.state[i]](&i, &myPlayers, &myAgents, msec);

		if (agentNodes[i] != NULL)
		{
			agentNodes[i]->SetPosition(Vector3(myAgents.x[i], myAgents.y[i], myAgents.z[i]));
		}
	}

	for (int i = 0; i < playerCount; ++i)
	{
		playerNodes[i]->SetPosition(Vector3(myPlayers.x[i], myPlayers.y[i], myPlayers.z[i]));
	}


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
	float dist = abs(pos.x - n.GetPosition().x);
	float sum = halfDim.x + aggroRange;

	if(dist <= sum) {
		dist = abs(pos.y - n.GetPosition().y);
		sum = halfDim.y + aggroRange;

		if(dist <= sum) {
			dist = abs(pos.z - n.GetPosition().z);
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
	myAgents.state[agentCount] = PATROLS; // starting state

	myAgents.targetLocation[agentCount] = 0;

	myAgents.patrolLocation[agentCount][0] = GenerateTargetLocation(a->GetPosition());	// start patrol
	myAgents.patrolLocation[agentCount][1] = GenerateTargetLocation(a->GetPosition());	// end patrol
	myAgents.patrolLocation[agentCount][2] = Vector3(0, 0, 0);							// store location

	myAgents.targetPlayer[agentCount] = -1; // no target player

	//abilities here

	myAgents.level[agentCount] = (rand() % 100) + 1; // randomly generate level

	myAgents.x[agentCount] = a->GetPosition().x; // store the x in an array
	myAgents.y[agentCount] = a->GetPosition().y; // store the y in an array
	myAgents.z[agentCount] = a->GetPosition().z; // store the z in an array

	agentNodes[agentCount] = a; // store the physic nodes for updating after cuda

	if (agentCount < Agents::MAXAGENTS - 1) // TODO probably should move this to the top
	{
		agentCount++;
	}
}

void AIManager::addPlayer(PhysicsNode* p)
{
	myPlayers.level[playerCount] = (rand() % 100) + 1; // randomly generate level

	myPlayers.hp[playerCount] = 1000; //set hp

	myPlayers.maxHP[playerCount] = 1000; //set the max hp

	myPlayers.isDead[playerCount] = false; // make the player alive

	myPlayers.x[playerCount] = p->GetPosition().x; // store the x in an array
	myPlayers.y[playerCount] = p->GetPosition().y; // store the y in an array
	myPlayers.z[playerCount] = p->GetPosition().z; // store the z in an array

	playerNodes[playerCount] = p;

	if (playerCount < Players::MAXPLAYERS - 1) // TODO probably should move this to the top
	{
		playerCount++;
	}

}