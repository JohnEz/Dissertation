#include "kernal.cuh"
#include <cmath>

/*__device__ template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}*/

__device__ void cudaPatrol(Players* players, Agents* agents, float msec, const unsigned int size)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	float MAXSPEED = 0.5F;

	//at target
	float diffX = agents->patrolLocation[a][agents->targetLocation[a]].x - agents->x[a];
	float diffZ = agents->patrolLocation[a][agents->targetLocation[a]].z - agents->z[a];
	float absX = abs(diffX);
	float absZ = abs(diffZ);

	//check its close enough to the point
	if (absX < 0.1f && absZ < 0.1f)
	{
		//get new target
		agents->targetLocation[a]++;
		agents->targetLocation[a] = agents->targetLocation[a] % 2; //need to fix this
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
		agents->x[a] += moveX * ((float(0) < diffX) - (diffX < float(0)));
		agents->z[a] += moveZ * ((float(0) < diffZ) - (diffZ < float(0)));
	}

	//state transition

	/*int i = 0;
	// loop through all the players
	while (i < players->MAXPLAYERS && agents->players[a][i] > -1)
	{

		//the player
		int p = agents->players[a][i];
		//calculate distance to player
		float3 diff = float3();

		diff.x = players->x[p] - agents->x[a];
		diff.y = players->y[p] - agents->y[a];
		diff.z = players->z[p] - agents->z[a];

		float dist = sqrtf((diff.x*diff.x)+(diff.y*diff.y)+(diff.z*diff.z));
	


		//if player close transition state to stare at player
		float aggroRange = min(agents->AGGRORANGE, agents->AGGRORANGE * ((float)agents->level[a] / (float)players->level[p]));

		if (dist < aggroRange && !players->isDead[p])
		{
			agents->state[a] = STARE_AT_PLAYER; //change state
			agents->patrolLocation[a][2].x = agents->x[a];
			agents->patrolLocation[a][2].y = agents->y[a];
			agents->patrolLocation[a][2].z = agents->z[a]; //set position it left patrol
			agents->targetPlayer[a] = p; // playing that is being stared at
			i = players->MAXPLAYERS; // exit the loop
		}
		i++;
	}*/
}

__device__ void cudaStareAtPlayer(Players* players, Agents* agents, float msec, const unsigned int size)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	int p = agents->targetPlayer[a]; // target player

	//calculate distance to player
	float3 diff = float3();
	diff.x = players->x[p] - agents->x[a];
	diff.y = players->y[p] - agents->y[a];
	diff.z = players->z[p] - agents->z[a];
	float dist = sqrtf((diff.x*diff.x)+(diff.y*diff.y)+(diff.z*diff.z));

	//the range of aggro, and pull, to the player
	float aggroRange = min(agents->AGGRORANGE, agents->AGGRORANGE * ((float)agents->level[a] / (float)players->level[p]));
	float pullRange = (aggroRange * 0.75f) * ((float)agents->level[a] / (float)players->level[p]);


	if (dist < pullRange && !players->isDead[p]) // if the player is in pull range
	{
		agents->state[a] = CHASE_PLAYER;
	}
	else
	{
		// if the player isnt in pull range check if there are any players closer
		bool playerClose = false;
		int i = 0;

		//loop through the players
		while (i < players->MAXPLAYERS && agents->players[a][i] > -1)
		{
			int p2 = agents->players[a][i];

			//calculate distance to player
			float3 diffNew = float3();
			diffNew.x = players->x[p2] - agents->x[a];
			diffNew.y = players->y[p2] - agents->y[a];
			diffNew.z = players->z[p2] - agents->z[a];
			float distNew = sqrtf((diffNew.x*diffNew.x)+(diffNew.y*diffNew.y)+(diffNew.z*diffNew.z));

			// if the new distance is less switch targte
			if (distNew <= dist  && !players->isDead[p2])
			{
				agents->targetPlayer[a] = p2;
				dist = distNew;
				float aggroRangeNew = min(agents->AGGRORANGE, agents->AGGRORANGE * (agents->level[a] / players->level[p2]));

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
			agents->state[a] = PATROL;
			agents->targetPlayer[a] = -1;
		}
	}
}

__device__ void cudaChasePlayer(Players* players, Agents* agents, float msec, const unsigned int size)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;
	
	float LEASHRANGE = 3200.0f;
	float ATTACKRANGE = 75.0f;
	float MAXSPEED = 0.5F;

	int p = agents->targetPlayer[a];

	//calculate distance to leash spot
	float3 diff = float3();
	diff.x = agents->patrolLocation[a][2].x - agents->x[a];
	diff.y = agents->patrolLocation[a][2].y - agents->y[a];
	diff.z = agents->patrolLocation[a][2].z - agents->z[a];

	float leashDist = sqrtf((diff.x*diff.x)+(diff.y*diff.y)+(diff.z*diff.z));;

	// if its too far away or if the player died leash back
	if (leashDist > LEASHRANGE || players->isDead[p])
	{
		agents->state[a] = LEASH;
		agents->targetPlayer[a] = -1;
	}
	else
	{
		//calculate distance to player
		float3 diff = float3();
		diff.x = players->x[p] - agents->x[a];
		diff.y = players->y[p] - agents->y[a];
		diff.z = players->z[p] - agents->z[a];
		float dist = sqrtf((diff.x*diff.x)+(diff.y*diff.y)+(diff.z*diff.z));

		//if close to player switch state to useability
		if (dist < ATTACKRANGE)
		{
			agents->state[a] = USE_ABILITY;
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
		agents->x[a] += moveX * ((float(0) < diff.x) - (diff.x < float(0)));
		agents->z[a] += moveZ * ((float(0) < diff.z) - (diff.z < float(0)));
	}

}

__device__ void cudaLeashBack(Players* players, Agents* agents, float msec, const unsigned int size)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	float MAXSPEED = 0.5F;

	//calculate distance to leash spot
	float diffX = agents->patrolLocation[a][2].x - agents->x[a];
	float diffZ = agents->patrolLocation[a][2].z - agents->z[a];
	float absX = abs(diffX);
	float absZ = abs(diffZ);

	//check its close enough to the point
	if (absX < 0.1f && absZ < 0.1f)
	{
		//change back to patrol
		agents->state[a] = PATROL;
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
		agents->x[a] += moveX * ((float(0) < diffX) - (diffX < float(0)));
		agents->z[a] += moveZ * ((float(0) < diffZ) - (diffZ < float(0)));
	}
}

__device__ void cudaUseAbility(Players* players, Agents* agents, float msec, const unsigned int size)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	float ATTACKRANGE = 75.0f;
	int p = agents->targetPlayer[a];

	if (players->isDead[p]) // if the player is dead
	{
		agents->state[a] = LEASH;	//leash back
		agents->targetPlayer[a] = -1; // set the target player to null
	}
	else
	{

		//TODO ADD ABILITIES BACK
		//look through abilities via priority until one is found not on cooldown
		int i = 0;
		while (i < agents->MAXABILITIES && agents->myAbilities[a][i].cooldown > 0.001f) {
			i++;
		}

		//cast ability
		if (i < agents->MAXABILITIES && agents->myAbilities[a][i].cooldown < 0.001f)
		{
			agents->myAbilities[a][i].cooldown = agents->myAbilities[a][i].maxCooldown;
			players->hp[agents->targetPlayer[a]] -= agents->myAbilities[a][i].damage;
		}

		//if the player goes out of range, change state to chase
		//calculate distance to player
		float3 diff = float3();
		diff.x = players->x[p] - agents->x[a];
		diff.y = players->y[p] - agents->y[a];
		diff.z = players->z[p] - agents->z[a];
		float dist = sqrtf((diff.x*diff.x)+(diff.y*diff.y)+(diff.z*diff.z));

		//if player close transition state to stare at player
		if (dist > (ATTACKRANGE))
		{
			agents->state[a] = CHASE_PLAYER;
		}
	}
}

__device__ void cudaBroadphase(Players* players, Agents* agents, float msec, const unsigned int size)
{

}

__global__ void cudaFSM(Players* players, Agents* agents, float msec, const unsigned int size)
{
	int a = blockIdx.x * blockDim.x + threadIdx.x;

	switch (agents->state[a]) {
	case PATROL: cudaPatrol(players, agents, msec, size);
		break;
	case STARE_AT_PLAYER: cudaStareAtPlayer(players, agents, msec, size);
		break;
	case CHASE_PLAYER: cudaChasePlayer(players, agents, msec, size);
		break;
	case LEASH: cudaLeashBack(players, agents, msec, size);
		break;
	case USE_ABILITY: cudaUseAbility(players, agents, msec, size);
		break;
	};

}



//put the data onto the GPU
cudaError_t addDataToGPU(Players* players, Agents* agents, Players* dev_players, Agents* dev_agents)
{
    cudaError_t cudaStatus;

	 // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

	// Allocate GPU buffers for three vectors (two input, one output)
	cudaStatus = cudaMalloc((void**)&dev_players, sizeof(Players));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed players!");
    }

	cudaStatus = cudaMalloc((void**)&dev_agents, sizeof(Agents));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed agents!");
    }


	// Copy input from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_players, players, sizeof(Players), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed players!");
    }

    cudaStatus = cudaMemcpy(dev_agents, agents, sizeof(Agents), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed agents!");
    }

	return cudaStatus;
}

cudaError_t runKernal(Players *dev_players, Agents *dev_agents, int agentCount, int msec, float* x, float* y, float* z)
{
	cudaError_t cudaStatus;

	//run patrol kernal
	//cudaPatrol<<<1, agentCount>>>(dev_agents);

	// Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

	// Copy output data from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(x, dev_agents->x, dev_agents->MAXAGENTS * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
       // fprintf(stderr, "cudaMemcpy failed getting Xs!");
    }

	cudaStatus = cudaMemcpy(y, dev_agents->y, dev_agents->MAXAGENTS * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        //fprintf(stderr, "cudaMemcpy failed getting Ys!");
    }

	cudaStatus = cudaMemcpy(z, dev_agents->z, dev_agents->MAXAGENTS * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        //fprintf(stderr, "cudaMemcpy failed getting Zs!");
    }

	cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }

	cudaFree(dev_agents);
    cudaFree(dev_players);

	return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(Players* players, Agents* agents, const unsigned int size, float msec)
{
    Players *dev_players = 0;
	Agents *dev_agents = 0;
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

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_agents, sizeof(Agents));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_players, sizeof(Players));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_agents, agents, sizeof(Agents), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_players, players, sizeof(Players), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaFSM, 0, size);

	// Round up according to array size 
	gridSize = (size + blockSize - 1) / blockSize;

	//cudaOccupancyMaxActiveBlocksPerMultiprocessor(&minGridSize, cudaPatrol, blockSize, 0);

	// Launch a kernel on the GPU with one thread for each element.
	cudaFSM<<<gridSize, blockSize>>>(dev_players, dev_agents, msec, size);

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

    // Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(players, dev_players, sizeof(Players), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed players!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(agents, dev_agents, sizeof(Agents), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed agents!");
        goto Error;
    }


Error:
    //cudaFree(dev_c);
    cudaFree(dev_agents);
    cudaFree(dev_players);
   

    return cudaStatus;
}