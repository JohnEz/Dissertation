#include "kernal.cuh"
#include <cmath>

/*__device__ template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}*/

__global__ void cudaPatrol(Players* players, Agents* agents, float msec)
{
	int a = threadIdx.x;

	float MAXSPEED = 0.5F;

	//at target
	float disX = agents->patrolLocation[a][agents->targetLocation[a]].x - agents->x[a];
	float disZ = agents->patrolLocation[a][agents->targetLocation[a]].z - agents->z[a];
	float absX = abs(disX);
	float absZ = abs(disZ);

	//check its close enough to the point
	if (absX < 10.1f && absZ < 10.1f)
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
		agents->x[a] += moveX * ((float(0) < disX) - (disX < float(0)));
		agents->z[a] += moveZ * ((float(0) < disZ) - (disZ < float(0)));
	}

	//state transition

	int i = 0;
	// loop through all the players
	/*while (i < Players::MAXPLAYERS && agents->players[a][i] > -1)
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
			agents->state[a] = STARE_AT_PLAYERS; //change state
			agents->patrolLocation[a][2].x = agents->x[a];
			agents->patrolLocation[a][2].y = agents->y[a];
			agents->patrolLocation[a][2].z = agents->z[a]; //set position it left patrol
			agents->targetPlayer[a] = p; // playing that is being stared at
			i = Player::MAX_PLAYERS; // exit the loop
		}
		i++;
	}*/
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
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
cudaError_t addWithCuda(Players* players, Agents* agents, unsigned int size, float msec)
{
    Players *dev_players = 0;
	Agents *dev_agents = 0;
    cudaError_t cudaStatus;

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

    cudaStatus = cudaMemcpy(dev_players, players, sizeof(Player), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	cudaPatrol<<<1, size>>>(dev_players, dev_agents, msec);

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
        fprintf(stderr, "cudaMemcpy failed X!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(agents, dev_agents, sizeof(Agents), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed Y!");
        goto Error;
    }


Error:
    //cudaFree(dev_c);
    cudaFree(dev_agents);
    cudaFree(dev_players);
   

    return cudaStatus;
}