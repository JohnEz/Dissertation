#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include "AIManager.h"

#ifndef CUDAFSM
#define CUDAFSM

//copies the data to the gpu memory
cudaError_t addDataToGPU(Players* players, Agents* agents, unsigned int size, float msec, Players* d_players, Agents* d_agents);

cudaError_t runKernal(Players* players, Agents* agents, unsigned int size, float msec, Players* d_players, Agents* d_agents);

cudaError_t clearData(Players* players, Agents* agents, unsigned int size, float msec, Players* d_players, Agents* d_agents);

cudaError_t addWithCuda(Players* players, Agents* agents, unsigned int size, float msec, Players* d_players, Agents* d_agents);

cudaError_t cudaUpdateAgents(Players* players, Agents* agents, const unsigned int size, float msec, AIWorldPartition* partitions, const int partitionCount, Vector3* halfDim);

#endif