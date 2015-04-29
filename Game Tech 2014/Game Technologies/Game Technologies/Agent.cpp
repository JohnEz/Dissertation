#include "Agent.h"

const float MAXSPEED = 1.0f;
const int PATROLSIZE = 3;
const float MAXAGGRORANGE = 400.0f;
const float ATTACKRANGE = 10.0f;
const float LEASHRANGE = 1600.0f;

void (*states[MAX_STATES]) (Player* players[], Player** targetPlayer, AgentState& state, int& currentTarget, Vector3 target[], PhysicsNode* physicsNode, float msec);


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

void Patrol(Player* players[], Player** targetPlayer, AgentState& state, int& currentTarget, Vector3 target[], PhysicsNode* pNode, float msec)
{
	//at target
	float disX = target[currentTarget].x - pNode->GetPosition().x;
	float disZ = target[currentTarget].z - pNode->GetPosition().z;
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

		Vector3 newPos = Vector3(pNode->GetPosition().x + (moveX * sgn<float>(disX)), pNode->GetPosition().y, pNode->GetPosition().z + (moveZ * sgn<float>(disZ)));

		pNode->SetPosition(newPos);
	}

	//state transition
	int i = 0;
	while (i < Player::MAX_PLAYERS && players[i] != NULL)
	{
		//calculate distance to player
		Vector3 diff = players[i]->physicsNode->GetPosition() - pNode->GetPosition();
		float dist = sqrtf(Vector3::Dot(diff, diff));

		//if player close transition state to stare at player
		if (dist < MAXAGGRORANGE) //add calculation including level
		{
			state = STARE_AT_PLAYER; //change state
			target[2] = pNode->GetPosition(); //set position it left patrol
			*targetPlayer = players[i]; // the player that is aggroing
			i = Player::MAX_PLAYERS; // exit the loop
		}
		i++;
	}

}

void stareAtPlayer(Player* players[], Player** targetPlayer, AgentState& state, int& currentTarget, Vector3 target[], PhysicsNode* pNode, float msec)
{
	//stop and face the player
	//if player gets closer statechange to chase player
	int i = 0;
	while (i < Player::MAX_PLAYERS && players[i] != NULL)
	{
		//calculate distance to player
		Vector3 diff = players[i]->physicsNode->GetPosition() - pNode->GetPosition();
		float dist = sqrtf(Vector3::Dot(diff, diff));

		//if player close transition state to stare at player
		if (dist < (MAXAGGRORANGE * 0.75f)) //add calculation including level
		{
			state = CHASE_PLAYER;
			i = Player::MAX_PLAYERS;
			*targetPlayer = players[i]; // the player that is aggroing
		}
		else if (dist > MAXAGGRORANGE) //add calculation including level
		{
			state = PATROL;
			i = Player::MAX_PLAYERS;
		}

		i++;
	}
	//if player gets further away resume patrol
}

void chasePlayer(Player* players[], Player** targetPlayer, AgentState& state, int& currentTarget, Vector3 target[], PhysicsNode* pNode, float msec)
{
	//calculate distance to leash spot
	Vector3 leashDiff = target[2] - pNode->GetPosition();
	float leashDist = sqrtf(Vector3::Dot(leashDiff, leashDiff));

	if (leashDist > LEASHRANGE)
	{
		state = LEASH;
	}
	else
	{
		//calculate distance to player
		Vector3 diff = (*targetPlayer)->physicsNode->GetPosition() - pNode->GetPosition();
		float dist = sqrtf(Vector3::Dot(diff, diff));

		//if close to player switch state to useability
		if (dist < ATTACKRANGE)
		{
			state = USE_ABILITY;
		}
	}


	//move towards players location
	float disX = (*targetPlayer)->physicsNode->GetPosition().x - pNode->GetPosition().x;
	float disZ = (*targetPlayer)->physicsNode->GetPosition().z - pNode->GetPosition().z;
	float absX = abs(disX);
	float absZ = abs(disZ);

	//move to target
	float dis = absX + absZ;
	float moveX = ((absX / dis) * MAXSPEED) * msec;
	float moveZ = ((absZ / dis) * MAXSPEED) * msec;

	moveX = min(moveX, absX);
	moveZ = min(moveZ, absZ);

	Vector3 newPos = Vector3(pNode->GetPosition().x + (moveX * sgn<float>(disX)), pNode->GetPosition().y, pNode->GetPosition().z + (moveZ * sgn<float>(disZ)));

	pNode->SetPosition(newPos);
}

void leashBack(Player* players[], Player** targetPlayer, AgentState& state, int& currentTarget, Vector3 target[], PhysicsNode* pNode, float msec)
{
	//at target
	float disX = target[2].x - pNode->GetPosition().x;
	float disZ = target[2].z - pNode->GetPosition().z;
	float absX = abs(disX);
	float absZ = abs(disZ);

	//check its close enough to the point (can lower the numbers when collision is removed)
	if (absX < 10.1f && absZ < 10.1f)
	{
		//change back to patrol
		state = PATROL;
	}
	else
	{
		//move to target
		float dis = absX + absZ;
		float moveX = ((absX / dis) * MAXSPEED) * msec;
		float moveZ = ((absZ / dis) * MAXSPEED) * msec;

		moveX = min(moveX, absX);
		moveZ = min(moveZ, absZ);

		Vector3 newPos = Vector3(pNode->GetPosition().x + (moveX * sgn<float>(disX)), pNode->GetPosition().y, pNode->GetPosition().z + (moveZ * sgn<float>(disZ)));

		pNode->SetPosition(newPos);
	}
}


Agent::Agent(SceneNode* s, PhysicsNode* p) {
	renderNode	= s;
	physicsNode = p;
	removed = false;

	myState = DEFUALT;
	subState = PATROL;
	states[PATROL] = Patrol;
	states[STARE_AT_PLAYER] = stareAtPlayer;
	states[CHASE_PLAYER] = chasePlayer;
	states[LEASH] = leashBack;

	targetLocation = 0;
	patrolLocations[0] = GenerateTargetLocation(physicsNode->GetPosition());  // start patrol
	patrolLocations[1] = GenerateTargetLocation(physicsNode->GetPosition());  // end patrol
	patrolLocations[2] = Vector3(0, 0, 0);			// store location


}

void Agent::Update(Player* players[], float msec)
{
	targetPlayer = players[0];
	//run the relevant state
	states[subState](players, &targetPlayer, subState, targetLocation, patrolLocations, physicsNode, msec);
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