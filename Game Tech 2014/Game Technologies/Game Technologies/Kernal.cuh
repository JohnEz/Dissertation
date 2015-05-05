#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include "AIManager.h"

#ifndef CUDAFSM
#define CUDAFSM

//copies the data to the gpu memory
cudaError_t addDataToGPU(Players* players, Agents* agents, Players* dev_players, Agents* dev_agents);

//runs patrol (at the moment)
cudaError_t runKernal(Players *dev_players, Agents *dev_agents, int agentCount, int msec, float* x, float* y, float* z);

cudaError_t addWithCuda(Players* players, Agents* agents, unsigned int size, float msec);

#endif