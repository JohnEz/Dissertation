#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include "AIManager.h"

#ifndef CUDAFSM
#define CUDAFSM



//copies the data to the gpu memory


cudaError_t cudaUpdateAgents(Players* players, Agents* agents, const unsigned int size, float msec, AIWorldPartition* partitions, const int partitionCount, Vector3* halfDim);

#endif