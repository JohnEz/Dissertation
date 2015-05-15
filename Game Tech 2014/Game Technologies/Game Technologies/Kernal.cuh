#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/version.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "AIManager.h"

#ifndef CUDAFSM
#define CUDAFSM

//copies the data to the gpu memory
cudaError_t cudaCopyCore(CopyOnce* coreData);

//Runs the FSM Kernal
cudaError_t cudaGPUBasic(CopyOnce* coreData, CopyEachFrame* updateData, const unsigned int agentCount, const unsigned int partitionCount, float msec);
cudaError_t cudaGPUCopyOnce(CopyOnce* coreData, CopyEachFrame* updateData, const unsigned int agentCount, const unsigned int partitionCount, float msec, bool runBroad);
cudaError_t cudaGPUBroad(CopyOnce* coreData, CopyEachFrame* updateData, const unsigned int agentCount, const unsigned int partitionCount, float msec, bool runBroad);
cudaError_t cudaGPUBroadAgents(CopyOnce* coreData, CopyEachFrame* updateData, const unsigned int agentCount, const unsigned int partitionCount, float msec, bool runBroad);
cudaError_t cudaGPUSplit(CopyOnce* coreData, CopyEachFrame* updateData, const unsigned int agentCount, const unsigned int partitionCount, float msec, bool runBroad);

cudaError_t cudaRunKernal(CopyOnce* coreData, CopyEachFrame* updateData, const unsigned int agentCount, const unsigned int partitionCount, float msec, bool runBroad);
cudaError_t cudaRunKernalDEBUG(CopyOnce* coreData, CopyEachFrame* updateData, const unsigned int agentCount, const unsigned int partitionCount, float msec, bool runBroad);
cudaError_t cudaRunKernalBase(CopyOnce* coreData, CopyEachFrame* updateData, const unsigned int agentCount, const unsigned int partitionCount, float msec, bool runBroad);
cudaError_t cudaRunKernalCleanWoSort(CopyOnce* coreData, CopyEachFrame* updateData, const unsigned int agentCount, const unsigned int partitionCount, float msec, bool runBroad);
cudaError_t cudaGPUSort(CopyOnce* coreData, CopyEachFrame* updateData, const unsigned int agentCount, const unsigned int partitionCount, float msec, bool runBroad);

//Gets the data from the GPU
cudaError_t copyDataFromGPU(CopyOnce* coreData, CopyEachFrame* updateData, const unsigned int agentCount, const unsigned int partitionCount, float msec);

//Clears the core data (dont forget to do this at the end!)
void clearCoreData();

cudaError_t cudaUpdateAgents(CopyOnce* coreData, CopyEachFrame* updateData, const unsigned int agentCount, const unsigned int partitionCount, float msec);

#endif