#include "Agent.h"

const float MAXSPEED = 1.0f;
const int PATROLSIZE = 3;

void (*states[MAX_STATES]) (AgentState& state, int& currentTarget, Vector3 target[], PhysicsNode* physicsNode, float msec);


template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

Vector3 GenerateTargetLocation()
{
	Vector3 target = Vector3(0,0,0);

	target.x = (rand() % 4000) - 2000;
	target.z = (rand() % 4000) - 2000;

	return target;
}

void Patrol(AgentState& state, int& currentTarget, Vector3 target[], PhysicsNode* physicsNode, float msec)
{
	//at target
	float disX = target[currentTarget].x - physicsNode->GetPosition().x;
	float disZ = target[currentTarget].z - physicsNode->GetPosition().z;
	float absX = abs(disX);
	float absZ = abs(disZ);

	//check its close enough to the point (can lower the numbers when collision is removed)
	if (absX < 10.1f && absZ < 10.1f)
	{
		//get new target
		currentTarget++;
		currentTarget = currentTarget % 2; //need to fix this
	}
	else
	{
		//move to target
		float dis = absX + absZ;
		float moveX = ((absX / dis) * MAXSPEED) * msec;
		float moveZ = ((absZ / dis) * MAXSPEED) * msec;

		moveX = min(moveX, absX);
		moveZ = min(moveZ, absZ);

		Vector3 newPos = Vector3(physicsNode->GetPosition().x + (moveX * sgn<float>(disX)), physicsNode->GetPosition().y, physicsNode->GetPosition().z + (moveZ * sgn<float>(disZ)));

		physicsNode->SetPosition(newPos);
	}

	//add state transitions here
	//if player close transition state to stare at player
}

void stareAtPlayer(AgentState& state, int& currentTarget, Vector3 target[], PhysicsNode* physicsNode, float msec)
{
	//stop and face the player
	//if player gets closer statechange to chase player
	//if player gets further away resume patrol
}

void chasePlayer(AgentState& state, int& currentTarget, Vector3 target[], PhysicsNode* physicsNode, float msec)
{
	//move towards players location
	//if close to player switch state to useability
	//if i am too far away from my patrol point, leash
}

Agent::Agent(SceneNode* s, PhysicsNode* p) {
	renderNode	= s;
	physicsNode = p;
	removed = false;

	myState = DEFUALT;
	subState = PATROL;
	states[PATROL] = Patrol;
	states[STARE_AT_PLAYER] = stareAtPlayer;

	targetLocation = 0;
	patrolLocations[0] = GenerateTargetLocation();  // start patrol
	patrolLocations[1] = GenerateTargetLocation();  // end patrol
	patrolLocations[2] = Vector3(0, 0, 0);			// store location
	

}

void Agent::Update(float msec)
{
	//run the relevant state
	states[subState](subState, targetLocation, patrolLocations, physicsNode, msec);
}

void Agent::ConnectToSystems() {
	if(renderNode) {
		Renderer::GetRenderer().AddNode(renderNode);
	}

	if(physicsNode) {
		PhysicsSystem::GetPhysicsSystem().AddNode(physicsNode);
	}

	if(renderNode && physicsNode) {
		physicsNode->SetTarget(renderNode);
	}
}

void Agent::DisconnectFromSystems() {
	if(renderNode) {
		Renderer::GetRenderer().RemoveNode(renderNode);
	}

	if(physicsNode) {
		PhysicsSystem::GetPhysicsSystem().RemoveNode(physicsNode);
	}
}