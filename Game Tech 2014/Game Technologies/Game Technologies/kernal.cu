#include "kernal.cuh"
#include <cmath>

//TODO CHANGE EVERYTHING TO POINTERS, WHY DIDNT I DO THAT IN THE FIRST PLACE
//TODO GET RID OFF ALL THE VECTOR3 TO FLOAT3 TRANSLATIONS

__device__ bool CheckBounding(float3 nPos, float aggroRange, float3 pos, float3 halfDim)
{
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

#pragma region TogetherStates

__device__ void cudaPatrol(CopyOnce* coreData, CopyEachFrame* updateData, short* agentsPartitions, short* partitionsPlayers, float msec)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	float MAXSPEED = 0.5F;

	//at target
	float diffX = coreData->myAgents.patrolLocation[a][coreData->myAgents.targetLocation[a]].x - coreData->myAgents.x[a];
	float diffZ = coreData->myAgents.patrolLocation[a][coreData->myAgents.targetLocation[a]].z - coreData->myAgents.z[a];
	float absX = abs(diffX);
	float absZ = abs(diffZ);

	//check its close enough to the point
	if (absX < 0.1f && absZ < 0.1f)
	{
		//get new target
		coreData->myAgents.targetLocation[a]++;
		coreData->myAgents.targetLocation[a] = coreData->myAgents.targetLocation[a] % 2; //need to fix this
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
		coreData->myAgents.x[a] += moveX * ((float(0) < diffX) - (diffX < float(0)));
		coreData->myAgents.z[a] += moveZ * ((float(0) < diffZ) - (diffZ < float(0)));
	}

	//state transition

	int i = 0;
	while (i < 8 && agentsPartitions[((a)*8) + i] != -1)
	{
		int j = 0;
		int part = agentsPartitions[(a*8) + i];
		int partPlayer = (part*coreData->myPlayers.MAXPLAYERS);
		while (j < coreData->myPlayers.MAXPLAYERS && partitionsPlayers[partPlayer+j] != -1)
		{
			//the player
			short p = partitionsPlayers[partPlayer+j];
			//calculate distance to player
			float3 diff = float3();

			diff.x = coreData->myPlayers.x[p] - coreData->myAgents.x[a];
			diff.y = coreData->myPlayers.y[p] - coreData->myAgents.y[a];
			diff.z = coreData->myPlayers.z[p] - coreData->myAgents.z[a];

			float dist = sqrtf((diff.x*diff.x)+(diff.y*diff.y)+(diff.z*diff.z));



			//if player close transition state to stare at player
			float aggroRange = min(coreData->myAgents.AGGRORANGE, coreData->myAgents.AGGRORANGE * ((float)coreData->myAgents.level[a] / (float)coreData->myPlayers.level[p]));

			if (dist < aggroRange && !updateData->playerIsDead[p])
			{
				coreData->myAgents.state[a] = STARE_AT_PLAYER; //change state
				coreData->myAgents.patrolLocation[a][2].x = coreData->myAgents.x[a];
				coreData->myAgents.patrolLocation[a][2].y = coreData->myAgents.y[a];
				coreData->myAgents.patrolLocation[a][2].z = coreData->myAgents.z[a]; //set position it left patrol
				coreData->myAgents.targetPlayer[a] = p; // playing that is being stared at
				i = coreData->myPlayers.MAXPLAYERS; // exit the loop
			}
			++j;
		}

		++i;
	}
}

__device__ void cudaStareAtPlayer(CopyOnce* coreData, CopyEachFrame* updateData, short* agentsPartitions, short* partitionsPlayers, float msec)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	int p = coreData->myAgents.targetPlayer[a]; // target player

	//calculate distance to player
	float3 diff = float3();
	diff.x = coreData->myPlayers.x[p] - coreData->myAgents.x[a];
	diff.y = coreData->myPlayers.y[p] - coreData->myAgents.y[a];
	diff.z = coreData->myPlayers.z[p] - coreData->myAgents.z[a];
	float dist = sqrtf((diff.x*diff.x)+(diff.y*diff.y)+(diff.z*diff.z));

	//the range of aggro, and pull, to the player
	float aggroRange = min(coreData->myAgents.AGGRORANGE, coreData->myAgents.AGGRORANGE * ((float)coreData->myAgents.level[a] / (float)coreData->myPlayers.level[p]));
	float pullRange = (aggroRange * 0.75f) * ((float)coreData->myAgents.level[a] / (float)coreData->myPlayers.level[p]);

	coreData->myAgents.waitedTime[a] += msec;

	if ((dist < pullRange || coreData->myAgents.waitedTime[a] > 8000.0f ) && !updateData->playerIsDead[p]) // if the player is in pull range
	{
		coreData->myAgents.state[a] = CHASE_PLAYER;
		coreData->myAgents.waitedTime[a] = 0.0f;
	}
	else
	{
		// if the player isnt in pull range check if there are any players closer
		bool playerClose = false;
		int i = 0;
		while (i < 8 && agentsPartitions[((a)*8) + i] != -1)
		{
			int j = 0;
			int part = agentsPartitions[(a*8) + i];
			int partPlayer = (part*coreData->myPlayers.MAXPLAYERS);
			while (j < coreData->myPlayers.MAXPLAYERS && partitionsPlayers[partPlayer+j] != -1)
			{
				//the player
				short p2 = partitionsPlayers[partPlayer+j];

				//calculate distance to player
				float3 diffNew = float3();
				diffNew.x = coreData->myPlayers.x[p2] - coreData->myAgents.x[a];
				diffNew.y = coreData->myPlayers.y[p2] - coreData->myAgents.y[a];
				diffNew.z = coreData->myPlayers.z[p2] - coreData->myAgents.z[a];
				float distNew = sqrtf((diffNew.x*diffNew.x)+(diffNew.y*diffNew.y)+(diffNew.z*diffNew.z));

				// if the new distance is less switch targte
				if (distNew <= dist  && !updateData->playerIsDead[p2])
				{
					coreData->myAgents.targetPlayer[a] = p2;
					dist = distNew;
					float aggroRangeNew = min(coreData->myAgents.AGGRORANGE, coreData->myAgents.AGGRORANGE * (coreData->myAgents.level[a] / coreData->myPlayers.level[p2]));

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
			coreData->myAgents.waitedTime[a] = 0.0f;
			coreData->myAgents.state[a] = PATROL;
			coreData->myAgents.targetPlayer[a] = -1;
		}
	}
}

__device__ void cudaChasePlayer(CopyOnce* coreData, CopyEachFrame* updateData, float msec)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	float LEASHRANGE = 3200.0f;
	float ATTACKRANGE = 75.0f;
	float MAXSPEED = 0.5F;

	int p = coreData->myAgents.targetPlayer[a];

	//calculate distance to leash spot
	float3 diff = float3();
	diff.x = coreData->myAgents.patrolLocation[a][2].x - coreData->myAgents.x[a];
	diff.y = coreData->myAgents.patrolLocation[a][2].y - coreData->myAgents.y[a];
	diff.z = coreData->myAgents.patrolLocation[a][2].z - coreData->myAgents.z[a];

	float leashDist = sqrtf((diff.x*diff.x)+(diff.y*diff.y)+(diff.z*diff.z));;

	// if its too far away or if the player died leash back
	if (leashDist > LEASHRANGE || updateData->playerIsDead[p])
	{
		coreData->myAgents.state[a] = LEASH;
		coreData->myAgents.targetPlayer[a] = -1;
	}
	else
	{
		//calculate distance to player
		float3 diff = float3();
		diff.x = coreData->myPlayers.x[p] - coreData->myAgents.x[a];
		diff.y = coreData->myPlayers.y[p] - coreData->myAgents.y[a];
		diff.z = coreData->myPlayers.z[p] - coreData->myAgents.z[a];
		float dist = sqrtf((diff.x*diff.x)+(diff.y*diff.y)+(diff.z*diff.z));

		//if close to player switch state to useability
		if (dist < ATTACKRANGE)
		{
			coreData->myAgents.state[a] = USE_ABILITY;
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
		coreData->myAgents.x[a] += moveX * ((float(0) < diff.x) - (diff.x < float(0)));
		coreData->myAgents.z[a] += moveZ * ((float(0) < diff.z) - (diff.z < float(0)));
	}

}

__device__ void cudaLeashBack(CopyOnce* coreData, CopyEachFrame* updateData, float msec)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	float MAXSPEED = 0.5F;

	//calculate distance to leash spot
	float diffX = coreData->myAgents.patrolLocation[a][2].x - coreData->myAgents.x[a];
	float diffZ = coreData->myAgents.patrolLocation[a][2].z - coreData->myAgents.z[a];
	float absX = abs(diffX);
	float absZ = abs(diffZ);

	//check its close enough to the point
	if (absX < 0.1f && absZ < 0.1f)
	{
		//change back to patrol
		coreData->myAgents.state[a] = PATROL;
	}
	else
	{
		//move to target
		float dis = absX + absZ;
		float moveX = ((absX / dis) * MAXSPEED) * msec;
		float moveZ = ((absZ / dis) * MAXSPEED) * msec;

		moveX = min(moveX, absX);
		moveZ = min(moveZ, absZ);

		//set new position
		coreData->myAgents.x[a] += moveX * ((float(0) < diffX) - (diffX < float(0)));
		coreData->myAgents.z[a] += moveZ * ((float(0) < diffZ) - (diffZ < float(0)));
	}
}

__device__ void cudaReduceCooldowns(CopyOnce* coreData, float msec)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0 ; i < coreData->myAgents.MAXABILITIES; ++i)
	{
		int check = coreData->myAgents.myAbilities[a][i].cooldown > 0;
		coreData->myAgents.myAbilities[a][i].cooldown -= msec * check;
	}
}

__device__ void cudaUseAbility(CopyOnce* coreData, CopyEachFrame* updateData, float msec)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	float ATTACKRANGE = 75.0f;
	int p = coreData->myAgents.targetPlayer[a];

	if (updateData->playerIsDead[p]) // if the player is dead
	{
		coreData->myAgents.state[a] = LEASH;	//leash back
		coreData->myAgents.targetPlayer[a] = -1; // set the target player to null
	}
	else
	{

		//TODO ADD ABILITIES BACK
		//look through abilities via priority until one is found not on cooldown
		int i = 0;
		while (i < coreData->myAgents.MAXABILITIES && coreData->myAgents.myAbilities[a][i].cooldown > 0.001f) {
			i++;
		}

		//cast ability
		if (i < coreData->myAgents.MAXABILITIES && coreData->myAgents.myAbilities[a][i].cooldown < 0.001f)
		{
			coreData->myAgents.myAbilities[a][i].cooldown = coreData->myAgents.myAbilities[a][i].maxCooldown;
			coreData->myPlayers.hp[coreData->myAgents.targetPlayer[a]] -= coreData->myAgents.myAbilities[a][i].damage;
		}

		//if the player goes out of range, change state to chase
		//calculate distance to player
		float3 diff = float3();
		diff.x = coreData->myPlayers.x[p] - coreData->myAgents.x[a];
		diff.y = coreData->myPlayers.y[p] - coreData->myAgents.y[a];
		diff.z = coreData->myPlayers.z[p] - coreData->myAgents.z[a];
		float dist = sqrtf((diff.x*diff.x)+(diff.y*diff.y)+(diff.z*diff.z));

		//if player close transition state to stare at player
		if (dist > (ATTACKRANGE))
		{
			coreData->myAgents.state[a] = CHASE_PLAYER;
		}
	}
}

__global__ void cudaFSM(CopyOnce* coreData, CopyEachFrame* updateData, short* agentsPartitions, short* partitionsPlayers, float msec)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	switch (coreData->myAgents.state[a]) {
	case PATROL: cudaPatrol(coreData, updateData, agentsPartitions, partitionsPlayers, msec);
		break;
	case STARE_AT_PLAYER: cudaStareAtPlayer(coreData, updateData, agentsPartitions, partitionsPlayers, msec);
		break;
	case CHASE_PLAYER: cudaChasePlayer(coreData, updateData, msec);
		break;
	case LEASH: cudaLeashBack(coreData, updateData, msec);
		break;
	case USE_ABILITY: cudaUseAbility(coreData, updateData, msec);
		break;
	};

	cudaReduceCooldowns(coreData, msec);

}

#pragma endregion

#pragma region seperatedStates

__device__ void cudaPatrolState(CopyOnce* coreData, CopyEachFrame* updateData, short* agentsPartitions, short* partitionsPlayers, float msec)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	float MAXSPEED = 0.5F;

	//at target
	float diffX = coreData->myAgents.patrolLocation[a][coreData->myAgents.targetLocation[a]].x - coreData->myAgents.x[a];
	float diffZ = coreData->myAgents.patrolLocation[a][coreData->myAgents.targetLocation[a]].z - coreData->myAgents.z[a];
	float absX = abs(diffX);
	float absZ = abs(diffZ);

	//check its close enough to the point
	if (absX < 0.1f && absZ < 0.1f)
	{
		//get new target
		coreData->myAgents.targetLocation[a]++;
		coreData->myAgents.targetLocation[a] = coreData->myAgents.targetLocation[a] % 2; //need to fix this
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
		coreData->myAgents.x[a] += moveX * ((float(0) < diffX) - (diffX < float(0)));
		coreData->myAgents.z[a] += moveZ * ((float(0) < diffZ) - (diffZ < float(0)));
	}	
}

__device__ void cudaPatrolTransitions(CopyOnce* coreData, CopyEachFrame* updateData, short* agentsPartitions, short* partitionsPlayers, float msec) {

	int a = blockIdx.x * blockDim.x + threadIdx.x;

	//state transition
	int i = 0;
	while (i < 8 && agentsPartitions[((a)*8) + i] != -1)
	{
		int j = 0;
		int part = agentsPartitions[(a*8) + i];
		int partPlayer = (part*coreData->myPlayers.MAXPLAYERS);
		while (j < coreData->myPlayers.MAXPLAYERS && partitionsPlayers[partPlayer+j] != -1)
		{
			//the player
			short p = partitionsPlayers[partPlayer+j];
			//calculate distance to player
			float3 diff = float3();

			diff.x = coreData->myPlayers.x[p] - coreData->myAgents.x[a];
			diff.y = coreData->myPlayers.y[p] - coreData->myAgents.y[a];
			diff.z = coreData->myPlayers.z[p] - coreData->myAgents.z[a];

			float dist = sqrtf((diff.x*diff.x)+(diff.y*diff.y)+(diff.z*diff.z));



			//if player close transition state to stare at player
			float aggroRange = min(coreData->myAgents.AGGRORANGE, coreData->myAgents.AGGRORANGE * ((float)coreData->myAgents.level[a] / (float)coreData->myPlayers.level[p]));

			if (dist < aggroRange && !updateData->playerIsDead[p])
			{
				coreData->myAgents.state[a] = STARE_AT_PLAYER; //change state
				coreData->myAgents.patrolLocation[a][2].x = coreData->myAgents.x[a];
				coreData->myAgents.patrolLocation[a][2].y = coreData->myAgents.y[a];
				coreData->myAgents.patrolLocation[a][2].z = coreData->myAgents.z[a]; //set position it left patrol
				coreData->myAgents.targetPlayer[a] = p; // playing that is being stared at
				i = coreData->myPlayers.MAXPLAYERS; // exit the loop
			}
			++j;
		}

		++i;
	}
}

__device__ void cudaStareAtPlayerState(CopyOnce* coreData, CopyEachFrame* updateData, short* agentsPartitions, short* partitionsPlayers, float msec)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	coreData->myAgents.waitedTime[a] += msec;
}

__device__ void cudaStareAtPlayerTransitions(CopyOnce* coreData, CopyEachFrame* updateData, short* agentsPartitions, short* partitionsPlayers, float msec)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	int p = coreData->myAgents.targetPlayer[a]; // target player

	//calculate distance to player
	float3 diff = float3();
	diff.x = coreData->myPlayers.x[p] - coreData->myAgents.x[a];
	diff.y = coreData->myPlayers.y[p] - coreData->myAgents.y[a];
	diff.z = coreData->myPlayers.z[p] - coreData->myAgents.z[a];
	float dist = sqrtf((diff.x*diff.x)+(diff.y*diff.y)+(diff.z*diff.z));

	//the range of aggro, and pull, to the player
	float aggroRange = min(coreData->myAgents.AGGRORANGE, coreData->myAgents.AGGRORANGE * ((float)coreData->myAgents.level[a] / (float)coreData->myPlayers.level[p]));
	float pullRange = (aggroRange * 0.75f) * ((float)coreData->myAgents.level[a] / (float)coreData->myPlayers.level[p]);

	if ((dist < pullRange || coreData->myAgents.waitedTime[a] > 8000.0f ) && !updateData->playerIsDead[p]) // if the player is in pull range
	{
		coreData->myAgents.state[a] = CHASE_PLAYER;
		coreData->myAgents.waitedTime[a] = 0.0f;
	}
	else
	{
		// if the player isnt in pull range check if there are any players closer
		bool playerClose = false;
		int i = 0;
		while (i < 8 && agentsPartitions[((a)*8) + i] != -1)
		{
			int j = 0;
			int part = agentsPartitions[(a*8) + i];
			int partPlayer = (part*coreData->myPlayers.MAXPLAYERS);
			while (j < coreData->myPlayers.MAXPLAYERS && partitionsPlayers[partPlayer+j] != -1)
			{
				//the player
				short p2 = partitionsPlayers[partPlayer+j];

				//calculate distance to player
				float3 diffNew = float3();
				diffNew.x = coreData->myPlayers.x[p2] - coreData->myAgents.x[a];
				diffNew.y = coreData->myPlayers.y[p2] - coreData->myAgents.y[a];
				diffNew.z = coreData->myPlayers.z[p2] - coreData->myAgents.z[a];
				float distNew = sqrtf((diffNew.x*diffNew.x)+(diffNew.y*diffNew.y)+(diffNew.z*diffNew.z));

				// if the new distance is less switch targte
				if (distNew <= dist  && !updateData->playerIsDead[p2])
				{
					coreData->myAgents.targetPlayer[a] = p2;
					dist = distNew;
					float aggroRangeNew = min(coreData->myAgents.AGGRORANGE, coreData->myAgents.AGGRORANGE * (coreData->myAgents.level[a] / coreData->myPlayers.level[p2]));

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
			coreData->myAgents.waitedTime[a] = 0.0f;
			coreData->myAgents.state[a] = PATROL;
			coreData->myAgents.targetPlayer[a] = -1;
		}
	}

}

__device__ void cudaChasePlayerState(CopyOnce* coreData, CopyEachFrame* updateData, float msec)
{
	float MAXSPEED = 0.5F;

	int a = blockIdx.x * blockDim.x + threadIdx.x;

	int p = coreData->myAgents.targetPlayer[a];

	//calculate distance to player
	float3 diff = float3();
	diff.x = coreData->myPlayers.x[p] - coreData->myAgents.x[a];
	diff.y = coreData->myPlayers.y[p] - coreData->myAgents.y[a];
	diff.z = coreData->myPlayers.z[p] - coreData->myAgents.z[a];

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
	coreData->myAgents.x[a] += moveX * ((float(0) < diff.x) - (diff.x < float(0)));
	coreData->myAgents.z[a] += moveZ * ((float(0) < diff.z) - (diff.z < float(0)));

}

__device__ void cudaChasePlayerTransitions(CopyOnce* coreData, CopyEachFrame* updateData, float msec) {
	float LEASHRANGE = 3200.0f;
	float ATTACKRANGE = 75.0f;

	int a = blockIdx.x * blockDim.x + threadIdx.x;

	int p = coreData->myAgents.targetPlayer[a];

	//calculate distance to leash spot
	float3 diff = float3();
	diff.x = coreData->myAgents.patrolLocation[a][2].x - coreData->myAgents.x[a];
	diff.y = coreData->myAgents.patrolLocation[a][2].y - coreData->myAgents.y[a];
	diff.z = coreData->myAgents.patrolLocation[a][2].z - coreData->myAgents.z[a];

	float leashDist = sqrtf((diff.x*diff.x)+(diff.y*diff.y)+(diff.z*diff.z));;

	// if its too far away or if the player died leash back
	if (leashDist > LEASHRANGE || updateData->playerIsDead[p])
	{
		coreData->myAgents.state[a] = LEASH;
		coreData->myAgents.targetPlayer[a] = -1;
	}
	else
	{
		//calculate distance to player
		float3 diff = float3();
		diff.x = coreData->myPlayers.x[p] - coreData->myAgents.x[a];
		diff.y = coreData->myPlayers.y[p] - coreData->myAgents.y[a];
		diff.z = coreData->myPlayers.z[p] - coreData->myAgents.z[a];
		float dist = sqrtf((diff.x*diff.x)+(diff.y*diff.y)+(diff.z*diff.z));

		//if close to player switch state to useability
		if (dist < ATTACKRANGE)
		{
			coreData->myAgents.state[a] = USE_ABILITY;
		}


	}
}

__device__ void cudaLeashBackState(CopyOnce* coreData, CopyEachFrame* updateData, float msec)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	float MAXSPEED = 0.5F;

	//calculate distance to leash spot
	float diffX = coreData->myAgents.patrolLocation[a][2].x - coreData->myAgents.x[a];
	float diffZ = coreData->myAgents.patrolLocation[a][2].z - coreData->myAgents.z[a];
	float absX = abs(diffX);
	float absZ = abs(diffZ);

	//move to target
	float dis = absX + absZ;
	float moveX = ((absX / dis) * MAXSPEED) * msec;
	float moveZ = ((absZ / dis) * MAXSPEED) * msec;

	moveX = min(moveX, absX);
	moveZ = min(moveZ, absZ);

	//set new position
	coreData->myAgents.x[a] += moveX * ((float(0) < diffX) - (diffX < float(0)));
	coreData->myAgents.z[a] += moveZ * ((float(0) < diffZ) - (diffZ < float(0)));

}

__device__ void cudaLeashBackTransitions(CopyOnce* coreData, CopyEachFrame* updateData, float msec) {

	int a = blockIdx.x * blockDim.x + threadIdx.x;

	//calculate distance to leash spot
	float diffX = coreData->myAgents.patrolLocation[a][2].x - coreData->myAgents.x[a];
	float diffZ = coreData->myAgents.patrolLocation[a][2].z - coreData->myAgents.z[a];
	float absX = abs(diffX);
	float absZ = abs(diffZ);

	//check its close enough to the point
	if (absX < 0.1f && absZ < 0.1f)
	{
		//change back to patrol
		coreData->myAgents.state[a] = PATROL;
	}

}

__device__ void cudaUseAbilityState(CopyOnce* coreData, CopyEachFrame* updateData, float msec)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	int p = coreData->myAgents.targetPlayer[a];

	//look through abilities via priority until one is found not on cooldown
	int i = 0;
	while (i < coreData->myAgents.MAXABILITIES && coreData->myAgents.myAbilities[a][i].cooldown > 0.001f) {
		++i;
	}

	//cast ability
	if (i < coreData->myAgents.MAXABILITIES && coreData->myAgents.myAbilities[a][i].cooldown < 0.001f)
	{
		coreData->myAgents.myAbilities[a][i].cooldown = coreData->myAgents.myAbilities[a][i].maxCooldown;
		coreData->myPlayers.hp[p] -= coreData->myAgents.myAbilities[a][i].damage;
	}

}

__device__ void cudaUseAbilityTransitions(CopyOnce* coreData, CopyEachFrame* updateData, float msec) {

	int a = blockIdx.x * blockDim.x + threadIdx.x;

	float ATTACKRANGE = 75.0f;
	int p = coreData->myAgents.targetPlayer[a];

	if (updateData->playerIsDead[p]) // if the player is dead
	{
		coreData->myAgents.state[a] = LEASH;	//leash back
		coreData->myAgents.targetPlayer[a] = -1; // set the target player to null
	}
	else
	{

		//if the player goes out of range, change state to chase
		//calculate distance to player
		float3 diff = float3();
		diff.x = coreData->myPlayers.x[p] - coreData->myAgents.x[a];
		diff.y = coreData->myPlayers.y[p] - coreData->myAgents.y[a];
		diff.z = coreData->myPlayers.z[p] - coreData->myAgents.z[a];
		float dist = sqrtf((diff.x*diff.x)+(diff.y*diff.y)+(diff.z*diff.z));

		//if player close transition state to stare at player
		if (dist > (ATTACKRANGE))
		{
			coreData->myAgents.state[a] = CHASE_PLAYER;
		}
	}

}

__global__ void cudaFSMSplit(CopyOnce* coreData, CopyEachFrame* updateData, short* agentsPartitions, short* partitionsPlayers, float msec)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	switch (coreData->myAgents.state[a]) {
	case PATROL: cudaPatrolTransitions(coreData, updateData, agentsPartitions, partitionsPlayers, msec);
		cudaPatrolState(coreData, updateData, agentsPartitions, partitionsPlayers, msec);
		break;
	case STARE_AT_PLAYER: cudaStareAtPlayerTransitions(coreData, updateData, agentsPartitions, partitionsPlayers, msec);
		cudaStareAtPlayerState(coreData, updateData, agentsPartitions, partitionsPlayers, msec);
		break;
	case CHASE_PLAYER: cudaChasePlayerTransitions(coreData, updateData, msec);
		cudaChasePlayerState(coreData, updateData, msec);
		break;
	case LEASH: cudaLeashBackTransitions(coreData, updateData, msec);
		cudaLeashBackState(coreData, updateData, msec);
		break;
	case USE_ABILITY: cudaUseAbilityTransitions(coreData, updateData, msec);
		cudaUseAbilityState(coreData, updateData, msec);
		break;
	};

	cudaReduceCooldowns(coreData, msec);

}

#pragma endregion

__global__ void cudaBroadphasePlayers(CopyOnce* coreData, CopyEachFrame* updateData, short* partitionsPlayers)
{
	int pa = blockIdx.x * blockDim.x + threadIdx.x;

	//position of this partition
	float3 pos = float3();
	pos.x = coreData->myPartitions.pos[pa].x;
	pos.y = coreData->myPartitions.pos[pa].y;
	pos.z = coreData->myPartitions.pos[pa].z;

	if (pos.x != 0 && pos.y != 0 && pos.z != 0)
	{
		updateData->playerCount[pa] = 0;

		// half dimensions of the partitions
		float3 halfDim = float3();
		halfDim.x = coreData->myPartitions.halfDim.x;
		halfDim.y = coreData->myPartitions.halfDim.y;
		halfDim.z = coreData->myPartitions.halfDim.z;

		//loop through the players
		for (int j = 0; j < coreData->myPlayers.MAXPLAYERS; ++j) {
			//check the player exists
			if(!updateData->playerIsDead[j] && coreData->myPlayers.maxHP[j] > 0)
			{

				//players position
				float3 playerPos = float3();
				playerPos.x = coreData->myPlayers.x[j];
				playerPos.y = coreData->myPlayers.y[j];
				playerPos.z = coreData->myPlayers.z[j];

				//check if its in the partition
				if (CheckBounding(playerPos, 0, pos, halfDim))
				{
					//add to the partitions players
					partitionsPlayers[(pa*coreData->myPlayers.MAXPLAYERS) + updateData->playerCount[pa]] = j;
					++updateData->playerCount[pa];
				}
			}
		}
	}

}

__global__ void cudaBroadphasePlayers2(CopyOnce* coreData, CopyEachFrame* updateData, short* partitionsPlayers, const int partitionCount)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;

	int part = t % partitionCount;
	int p = t / partitionCount;

	if (p == 0) updateData->playerCount[part] = 0;
	__syncthreads();

	if(!updateData->playerIsDead[p] && coreData->myPlayers.maxHP[p] > 0) {

		//players position
		float3 playerPos = float3();
		playerPos.x = coreData->myPlayers.x[p];
		playerPos.y = coreData->myPlayers.y[p];
		playerPos.z = coreData->myPlayers.z[p];

		// half dimensions of the partitions
		float3 halfDim = float3();
		halfDim.x = coreData->myPartitions.halfDim.x;
		halfDim.y = coreData->myPartitions.halfDim.y;
		halfDim.z = coreData->myPartitions.halfDim.z;

		//position of this partition
		float3 partPos = float3();
		partPos.x = coreData->myPartitions.pos[part].x;
		partPos.y = coreData->myPartitions.pos[part].y;
		partPos.z = coreData->myPartitions.pos[part].z;

		//check if its in the partition
		if (CheckBounding(playerPos, 0, partPos, halfDim))
		{
			//add to the partitions players
			int t = (part*coreData->myPlayers.MAXPLAYERS) + atomicAdd(&updateData->playerCount[part], 1);
			__syncthreads();
			partitionsPlayers[t] = p;
		}
	}

}

__global__ void cudaBroadphasePlayersCondence(CopyOnce* coreData, CopyEachFrame* updateData, short* partitionsPlayers, int partitionCount)
{
	int part = blockIdx.x * blockDim.x + threadIdx.x;

	updateData->playerCount[part] = 0;

	for (int i = 0; i < coreData->myPlayers.MAXPLAYERS; ++i)
	{
		if (partitionsPlayers[(part*coreData->myPlayers.MAXPLAYERS) + i] != -1)
		{
			partitionsPlayers[(part*coreData->myPlayers.MAXPLAYERS) + updateData->playerCount[part]] =  partitionsPlayers[(part*coreData->myPlayers.MAXPLAYERS) + i];
			++updateData->playerCount[part];
		}
	}
}

__global__ void cudaBroadphaseAgents(CopyOnce* coreData, CopyEachFrame* updateData,short* agentsPartitions, const int partitionCount)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	float3 agentPos = float3();
	agentPos.x = coreData->myAgents.x[a];
	agentPos.y = coreData->myAgents.y[a];
	agentPos.z = coreData->myAgents.z[a];

	// half dimensions of the partitions
	float3 halfDim = float3();
	halfDim.x = coreData->myPartitions.halfDim.x;
	halfDim.y = coreData->myPartitions.halfDim.y;
	halfDim.z = coreData->myPartitions.halfDim.z;

	int p = 0;

	//loop through the world partitions
	for (int j = 0; j < partitionCount; ++j) {

		//position of this partition
		float3 pos = float3();
		pos.x = coreData->myPartitions.pos[j].x;
		pos.y = coreData->myPartitions.pos[j].y;
		pos.z = coreData->myPartitions.pos[j].z;

		//check if the agent is in this partition
		if (pos.x != 0 && pos.y != 0 && pos.z != 0){

			if (CheckBounding(agentPos, coreData->myAgents.AGGRORANGE, pos, halfDim))
			{
				agentsPartitions[(a*8) + p] = j;
				++p;
			}
		}
	}
}

__global__ void cudaBroadphaseAgents2(CopyOnce* coreData, CopyEachFrame* updateData,short* agentsPartitions, const int partitionCount)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;

	int part = t % partitionCount;
	int a = t / partitionCount;

	float3 agentPos = float3();
	agentPos.x = coreData->myAgents.x[a];
	agentPos.y = coreData->myAgents.y[a];
	agentPos.z = coreData->myAgents.z[a];

	// half dimensions of the partitions
	float3 halfDim = float3();
	halfDim.x = coreData->myPartitions.halfDim.x;
	halfDim.y = coreData->myPartitions.halfDim.y;
	halfDim.z = coreData->myPartitions.halfDim.z;

	if (part == 0) coreData->myAgents.partCount[a] = 0;
	__syncthreads();

	//position of this partition
	float3 pos = float3();
	pos.x = coreData->myPartitions.pos[part].x;
	pos.y = coreData->myPartitions.pos[part].y;
	pos.z = coreData->myPartitions.pos[part].z;

	//check if the agent is in this partition
	if (pos.x != 0 && pos.y != 0 && pos.z != 0){

		if (CheckBounding(agentPos, coreData->myAgents.AGGRORANGE, pos, halfDim))
		{
			int v = (a*8) + atomicAdd(&coreData->myAgents.partCount[a], 1);
			__syncthreads();
			agentsPartitions[v] = part;
		}
	}
}

cudaError_t cudaUpdateAgents(CopyOnce* coreData, CopyEachFrame* updateData, const unsigned int agentCount, const unsigned int partitionCount, float msec)
{
	//COPY DATA TO THE GPU
	//////////////////////

	CopyOnce* d_coreData = 0;
	CopyEachFrame* d_updateData = 0;
	short* d_agentPartitions = 0;
	short* d_partitionPlayers = 0;
	cudaError_t cudaStatus;
	int blockSize;      // The launch configurator returned block size 
	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
	int gridSize;       // The actual grid size needed, based on input size 


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for the data
	// Agents
	cudaStatus = cudaMalloc((void**)&d_coreData, sizeof(CopyOnce));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Players
	cudaStatus = cudaMalloc((void**)&d_updateData, sizeof(CopyEachFrame));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// copy pointers from structs (urgh)
	cudaStatus = cudaMalloc((void**)&d_agentPartitions, (coreData->myAgents.MAXAGENTS*8) * sizeof(short));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_partitionPlayers, (coreData->myPartitions.MAXPARTITIONS*coreData->myPlayers.MAXPLAYERS) * sizeof(short));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}



	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(d_coreData, coreData, sizeof(CopyOnce), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_updateData, updateData, sizeof(CopyEachFrame), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_agentPartitions, updateData->agentPartitions, (coreData->myAgents.MAXAGENTS*8) * sizeof(short), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_partitionPlayers, updateData->partitionPlayers, (coreData->myPartitions.MAXPARTITIONS*coreData->myPlayers.MAXPLAYERS) * sizeof(short), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//RUN THE KERNALS ON THE GPU
	////////////////////////////

	//get the mingrid and blocksize
	/*cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaBroadphasePlayers, 0, partitionCount);

	// Round up according to array size 
	gridSize = (size + blockSize - 1) / blockSize;

	// Launch a kernel on the GPU with one thread for each element.
	cudaBroadphasePlayers<<<gridSize, blockSize>>>(d_players, d_partitions, partitionCount, halfDim);

	cudaStatus = cudaDeviceSynchronize();
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	goto Error;
	}

	///////////////////////////

	//get the mingrid and blocksize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaBroadphaseAgents, 0, size);

	// Round up according to array size 
	gridSize = (size + blockSize - 1) / blockSize;

	// Launch a kernel on the GPU with one thread for each element.
	cudaBroadphaseAgents<<<gridSize, blockSize>>>(d_agents, d_partitions, partitionCount, halfDim);

	cudaStatus = cudaDeviceSynchronize();
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	goto Error;
	}*/

	//////////////////////////

	//get the mingrid and blocksize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaFSM, 0, agentCount);

	// Round up according to array size 
	gridSize = (agentCount + blockSize - 1) / blockSize;

	// Launch a kernel on the GPU with one thread for each element.
	cudaFSM<<<gridSize, blockSize>>>(d_coreData, d_updateData, d_agentPartitions, d_partitionPlayers, msec);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}


	//COPY THE DATA BACK OFF OF THE GPU
	///////////////////////////////////

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(coreData, d_coreData, sizeof(CopyOnce), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed CoreData!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(updateData, d_updateData, sizeof(CopyEachFrame), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed UpdateData!");
		goto Error;
	}


Error:
	cudaFree(d_coreData);
	cudaFree(d_updateData);
	cudaFree(d_agentPartitions);
	cudaFree(d_partitionPlayers);

	return cudaStatus;
}

cudaError_t cudaCopyCore(CopyOnce* coreData)
{
	cudaError_t cudaStatus;

	//COPY DATA TO THE GPU
	//////////////////////

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	// Allocate GPU buffers for the data
	// CoreData
	cudaStatus = cudaMalloc((void**)&AIManager::GetInstance()->d_coreData, sizeof(CopyOnce));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		fprintf(stderr, cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}


	// Copy data to the gpu.
	cudaStatus = cudaMemcpy(AIManager::GetInstance()->d_coreData, coreData, sizeof(CopyOnce), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	return cudaStatus;
}

cudaError_t cudaRunKernal(CopyOnce* coreData, CopyEachFrame* updateData, const unsigned int agentCount, const unsigned int partitionCount, float msec, bool runBroad)
{
	AIManager::GetInstance()->d_updateData = 0;
	short* d_agentPartitions = 0;
	short* d_partitionPlayers = 0;
	cudaError_t cudaStatus;
	int blockSize;      // The launch configurator returned block size 
	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
	int gridSize;       // The actual grid size needed, based on input size

	//COPY THE NEW DATA TO THE GPU
	//////////////////////////////

	// Update Data
	cudaStatus = cudaMalloc((void**)&AIManager::GetInstance()->d_updateData, sizeof(CopyEachFrame));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}

	// Pointer Data
	cudaStatus = cudaMalloc((void**)&d_agentPartitions, (coreData->myAgents.MAXAGENTS*8) * sizeof(short));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}

	// Pointer Data
	cudaStatus = cudaMalloc((void**)&d_partitionPlayers, (coreData->myPartitions.MAXPARTITIONS*coreData->myPlayers.MAXPLAYERS) * sizeof(short));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}


	//Pass pointers their data
	//Update Data
	cudaStatus = cudaMemcpy(AIManager::GetInstance()->d_updateData, updateData, sizeof(CopyEachFrame), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	//Agent's Partitions
	cudaStatus = cudaMemcpy(d_agentPartitions, updateData->agentPartitions, (coreData->myAgents.MAXAGENTS*8) * sizeof(short), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	//Partition's Players
	cudaStatus = cudaMemcpy(d_partitionPlayers, updateData->partitionPlayers, (coreData->myPartitions.MAXPARTITIONS*coreData->myPlayers.MAXPLAYERS) * sizeof(short), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	//RUN THE KERNALS ON THE GPU
	////////////////////////////

	//run the broadphase on the gpu
	if (runBroad)
	{
		//BROADPHASE FOR PLAYERS
		////////////////////////

		//get the mingrid and blocksize
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaBroadphasePlayers, 0, partitionCount);

		// Round up according to array size 
		gridSize = (partitionCount + blockSize - 1) / blockSize;

		cudaBroadphasePlayers<<<gridSize, blockSize>>>(AIManager::GetInstance()->d_coreData, AIManager::GetInstance()->d_updateData, d_partitionPlayers);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "1st cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			fprintf(stderr, cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		//BROADPHASE FOR AGENTS
		///////////////////////

		//get the mingrid and blocksize
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaBroadphaseAgents, 0, agentCount);

		// Round up according to array size 
		gridSize = (agentCount + blockSize - 1) / blockSize;

		cudaBroadphaseAgents<<<gridSize, blockSize>>>(AIManager::GetInstance()->d_coreData, AIManager::GetInstance()->d_updateData, d_agentPartitions, partitionCount);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "2nd cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			fprintf(stderr, cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
	}

	//get the mingrid and blocksize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaFSM, 0, agentCount);

	// Round up according to array size 
	gridSize = (agentCount + blockSize - 1) / blockSize;

	// Launch a kernel on the GPU with one thread for each element.
	cudaFSM<<<gridSize, blockSize>>>(AIManager::GetInstance()->d_coreData, AIManager::GetInstance()->d_updateData, d_agentPartitions, d_partitionPlayers, msec);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "3rd cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		fprintf(stderr, cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	//clear partition data as we dont need to copy it back //POTENTIAL PROBLEM HERE
	cudaFree(d_agentPartitions);
	cudaFree(d_partitionPlayers);

	return cudaStatus;
}

cudaError_t cudaRunKernalNew(CopyOnce* coreData, CopyEachFrame* updateData, const unsigned int agentCount, const unsigned int partitionCount, float msec, bool runBroad)
{
	AIManager::GetInstance()->d_updateData = 0;
	short* d_agentPartitions = 0;
	short* d_partitionPlayers = 0;
	cudaError_t cudaStatus;
	int blockSize;      // The launch configurator returned block size 
	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
	int gridSize;       // The actual grid size needed, based on input size

	//COPY THE NEW DATA TO THE GPU
	//////////////////////////////

	// Update Data
	cudaStatus = cudaMalloc((void**)&AIManager::GetInstance()->d_updateData, sizeof(CopyEachFrame));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}

	// Pointer Data
	cudaStatus = cudaMalloc((void**)&d_agentPartitions, (coreData->myAgents.MAXAGENTS*8) * sizeof(short));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}

	// Pointer Data
	cudaStatus = cudaMalloc((void**)&d_partitionPlayers, (coreData->myPartitions.MAXPARTITIONS*coreData->myPlayers.MAXPLAYERS) * sizeof(short));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}


	//Pass pointers their data
	//Update Data
	cudaStatus = cudaMemcpy(AIManager::GetInstance()->d_updateData, updateData, sizeof(CopyEachFrame), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	//Agent's Partitions
	cudaStatus = cudaMemcpy(d_agentPartitions, updateData->agentPartitions, (coreData->myAgents.MAXAGENTS*8) * sizeof(short), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	//Partition's Players
	cudaStatus = cudaMemcpy(d_partitionPlayers, updateData->partitionPlayers, (coreData->myPartitions.MAXPARTITIONS*coreData->myPlayers.MAXPLAYERS) * sizeof(short), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	//RUN THE KERNALS ON THE GPU
	////////////////////////////

	//run the broadphase on the gpu
	if (runBroad)
	{
		//BROADPHASE FOR PLAYERS
		////////////////////////

		//get the mingrid and blocksize
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaBroadphasePlayers, 0, partitionCount*coreData->myPlayers.MAXPLAYERS);

		// Round up according to array size 
		gridSize = (partitionCount*coreData->myPlayers.MAXPLAYERS + blockSize - 1) / blockSize;

		cudaBroadphasePlayers2<<<gridSize, blockSize>>>(AIManager::GetInstance()->d_coreData, AIManager::GetInstance()->d_updateData, d_partitionPlayers, partitionCount);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "1st cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			fprintf(stderr, cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		//get the mingrid and blocksize
		/*cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaBroadphasePlayers, 0, partitionCount);

		// Round up according to array size 
		gridSize = (partitionCount + blockSize - 1) / blockSize;

		cudaBroadphasePlayersCondence<<<gridSize, blockSize>>>(AIManager::GetInstance()->d_coreData, AIManager::GetInstance()->d_updateData, d_partitionPlayers, partitionCount);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "1st cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		fprintf(stderr, cudaGetErrorString(cudaStatus));
		return cudaStatus;
		}*/

		//BROADPHASE FOR AGENTS
		///////////////////////

		//get the mingrid and blocksize
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaBroadphaseAgents, 0, agentCount*partitionCount);
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaBroadphaseAgents, 0, agentCount);

		// Round up according to array size 
		//gridSize = (agentCount*partitionCount + blockSize - 1) / blockSize;
		gridSize = (agentCount + blockSize - 1) / blockSize;

		cudaBroadphaseAgents<<<gridSize, blockSize>>>(AIManager::GetInstance()->d_coreData, AIManager::GetInstance()->d_updateData, d_agentPartitions, partitionCount);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "2nd cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			fprintf(stderr, cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
	}

	//get the mingrid and blocksize
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaFSM, 0, agentCount);

	// Round up according to array size 
	gridSize = (agentCount + blockSize - 1) / blockSize;

	// Launch a kernel on the GPU with one thread for each element.
	cudaFSMSplit<<<gridSize, blockSize>>>(AIManager::GetInstance()->d_coreData, AIManager::GetInstance()->d_updateData, d_agentPartitions, d_partitionPlayers, msec);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "3rd cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		fprintf(stderr, cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	//clear partition data as we dont need to copy it back //POTENTIAL PROBLEM HERE
	cudaFree(d_agentPartitions);
	cudaFree(d_partitionPlayers);

	return cudaStatus;
}

cudaError_t copyDataFromGPU(CopyOnce* coreData, CopyEachFrame* updateData, const unsigned int agentCount, const unsigned int partitionCount, float msec)
{
	cudaError_t cudaStatus;

	//COPY THE DATA BACK OFF OF THE GPU
	///////////////////////////////////

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(coreData, AIManager::GetInstance()->d_coreData, sizeof(CopyOnce), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed CoreData!");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(updateData, AIManager::GetInstance()->d_updateData, sizeof(CopyEachFrame), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed UpdateData!");
		return cudaStatus;
	}

	//clear updateData so we can send it again
	cudaFree(AIManager::GetInstance()->d_updateData);

	return cudaStatus;
}

void clearCoreData()
{
	cudaFree(AIManager::GetInstance()->d_coreData);
}