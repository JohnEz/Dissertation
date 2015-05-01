#include "Agent.h"

const float MAXSPEED = 0.5f;
const int PATROLSIZE = 3;
const int MAXABILITIES = 3;
const float Agent::MAXAGGRORANGE = 1000.0f;
const float ATTACKRANGE = 75.0f;
const float LEASHRANGE = 3200.0f;

//void (*states[MAX_STATES]) (Player* players[], Player& targetPlayer, AgentState& state, int& currentTarget, Vector3* target[], PhysicsNode& pNode, Ability* abilities[], Agent& a, float msec);
void (*states[MAX_STATES]) (Player* players[], Agent& a, float msec);


template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

Vector3* GenerateTargetLocation(const Vector3& position)
{
	Vector3* target = new Vector3;

	*target = position;

	target->x += (rand() % 4000) - 2000;
	target->z += (rand() % 4000) - 2000;

	return target;
}

void Patrol(Player* players[], Agent& a, float msec)
{
	//at target
	float disX = a.patrolLocations[a.targetLocation]->x - a.physicsNode->GetPosition().x;
	float disZ = a.patrolLocations[a.targetLocation]->z - a.physicsNode->GetPosition().z;
	float absX = abs(disX);
	float absZ = abs(disZ);

	//check its close enough to the point
	if (absX < 0.1f && absZ < 0.1f)
	{
		//get new target
		a.targetLocation++;
		a.targetLocation = a.targetLocation % 2; //need to fix this
	}
	else
	{
		//move to target
		float dis = absX + absZ;
		float moveX = ((absX / dis) * MAXSPEED) * msec;
		float moveZ = ((absZ / dis) * MAXSPEED) * msec;

		moveX = min(moveX, absX);
		moveZ = min(moveZ, absZ);

		Vector3 newPos = Vector3(a.physicsNode->GetPosition().x + (moveX * sgn<float>(disX)), a.physicsNode->GetPosition().y, a.physicsNode->GetPosition().z + (moveZ * sgn<float>(disZ)));

		a.physicsNode->SetPosition(newPos);
	}

	//state transition
	int i = 0;
	while (i < Player::MAX_PLAYERS && players[i] != NULL)
	{
		//calculate distance to player
		Vector3 diff = players[i]->physicsNode->GetPosition() - a.physicsNode->GetPosition();
		float dist = sqrtf(Vector3::Dot(diff, diff));

		//if player close transition state to stare at player
		if (dist < Agent::MAXAGGRORANGE * (a.level / players[i]->level) && !players[i]->isDead)
		{
			a.subState = STARE_AT_PLAYER; //change state
			*a.patrolLocations[2] = a.physicsNode->GetPosition(); //set position it left patrol
			a.targetPlayer = players[i]; // playing that is being stared at
			i = Player::MAX_PLAYERS; // exit the loop
		}
		i++;
	}

}

void stareAtPlayer(Player* players[], Agent& a, float msec)
{
	//calculate distance to player
	Vector3 diff = a.targetPlayer->physicsNode->GetPosition() - a.physicsNode->GetPosition();
	float dist = sqrtf(Vector3::Dot(diff, diff));

	if (dist < (Agent::MAXAGGRORANGE * 0.75f) * (a.level / a.targetPlayer->level) && !a.targetPlayer->isDead) // if the player is in pull range
	{
		a.subState = CHASE_PLAYER;
	}
	else
	{
		bool playerClose = false;
		int i = 0;

		while (i < Player::MAX_PLAYERS && players[i] != NULL && !players[i]->isDead)
		{
			//calculate distance to player
			Vector3 diffNew = players[i]->physicsNode->GetPosition() - a.physicsNode->GetPosition();
			float distNew = sqrtf(Vector3::Dot(diffNew, diffNew));

			if (distNew <= dist)
			{
				a.targetPlayer = players[i];
				dist = distNew;
				if (dist < Agent::MAXAGGRORANGE * (a.level / players[i]->level))
				{
					playerClose = true;
				}
			}
			++i;
		}

		if (!playerClose)
		{
			a.subState = PATROL;
			a.targetPlayer = NULL;
		}
	}
}

void chasePlayer(Player* players[], Agent& a, float msec)
{
	//calculate distance to leash spot
	Vector3 leashDiff = *a.patrolLocations[2] - a.physicsNode->GetPosition();
	float leashDist = sqrtf(Vector3::Dot(leashDiff, leashDiff));

	if (leashDist > LEASHRANGE || a.targetPlayer->isDead)
	{
		a.subState = LEASH;
		a.targetPlayer = NULL;
	}
	else
	{
		//calculate distance to player
		Vector3 diff = a.targetPlayer->physicsNode->GetPosition() - a.physicsNode->GetPosition();
		float dist = sqrtf(Vector3::Dot(diff, diff));

		//if close to player switch state to useability
		if (dist < ATTACKRANGE)
		{
			a.subState = USE_ABILITY;
		}

		//move towards players location
		float disX = a.targetPlayer->physicsNode->GetPosition().x - a.physicsNode->GetPosition().x;
		float disZ = a.targetPlayer->physicsNode->GetPosition().z - a.physicsNode->GetPosition().z;
		float absX = abs(disX);
		float absZ = abs(disZ);

		//move to target
		float dis = absX + absZ;
		float moveX = ((absX / dis) * MAXSPEED) * msec;
		float moveZ = ((absZ / dis) * MAXSPEED) * msec;

		moveX = min(moveX, absX);
		moveZ = min(moveZ, absZ);

		Vector3 newPos = Vector3(a.physicsNode->GetPosition().x + (moveX * sgn<float>(disX)), a.physicsNode->GetPosition().y, a.physicsNode->GetPosition().z + (moveZ * sgn<float>(disZ)));

		a.physicsNode->SetPosition(newPos);
	}

}

void leashBack(Player* players[], Agent& a, float msec)
{
	//at target
	float disX = a.patrolLocations[2]->x - a.physicsNode->GetPosition().x;
	float disZ = a.patrolLocations[2]->z - a.physicsNode->GetPosition().z;
	float absX = abs(disX);
	float absZ = abs(disZ);

	//check its close enough to the point
	if (absX < 0.1f && absZ < 0.1f)
	{
		//change back to patrol
		a.subState = PATROL;
	}
	else
	{
		//move to target
		float dis = absX + absZ;
		float moveX = ((absX / dis) * MAXSPEED) * msec;
		float moveZ = ((absZ / dis) * MAXSPEED) * msec;

		moveX = min(moveX, absX);
		moveZ = min(moveZ, absZ);

		Vector3 newPos = Vector3(a.physicsNode->GetPosition().x + (moveX * sgn<float>(disX)), a.physicsNode->GetPosition().y, a.physicsNode->GetPosition().z + (moveZ * sgn<float>(disZ)));

		a.physicsNode->SetPosition(newPos);
	}
}

void useAbility(Player* players[], Agent& a, float msec)
{
	if (a.targetPlayer->isDead)
	{
		a.subState = LEASH;
		a.targetPlayer = NULL;
	}
	else
	{
		//look through abilities via priority until one is found not on cooldown
		int i = 0;
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
		}

		//if the player goes out of range, change state to chase
		//calculate distance to player
		Vector3 diff = a.targetPlayer->physicsNode->GetPosition() - a.physicsNode->GetPosition();
		float dist = sqrtf(Vector3::Dot(diff, diff));

		//if player close transition state to stare at player
		if (dist > (ATTACKRANGE))
		{
			a.subState = CHASE_PLAYER;
		}
	}
}

Agent::Agent(SceneNode* s, PhysicsNode* p) {
	renderNode	= s;
	physicsNode = p;
	removed = false;

	subState = PATROL;
	states[PATROL] = Patrol;
	states[STARE_AT_PLAYER] = stareAtPlayer;
	states[CHASE_PLAYER] = chasePlayer;
	states[LEASH] = leashBack;
	states[USE_ABILITY] = useAbility;

	targetLocation = 0;
	patrolLocations[0] = GenerateTargetLocation(physicsNode->GetPosition());	// start patrol
	patrolLocations[1] = GenerateTargetLocation(physicsNode->GetPosition());	// end patrol
	patrolLocations[2] = new Vector3(0, 0, 0);									// store location

	targetPlayer = NULL;

	myAbilities[0] = new Ability();
	myAbilities[0]->maxCooldown = 20000.0f;
	myAbilities[0]->damage = 240;
	myAbilities[0]->targetEnemy = true;

	myAbilities[1] = new Ability();
	myAbilities[1]->maxCooldown = 14000.0f;
	myAbilities[1]->damage = 140;
	myAbilities[1]->targetEnemy = true;

	myAbilities[2] = new Ability();
	myAbilities[2]->maxCooldown = 1000.0f;
	myAbilities[2]->damage = 9;
	myAbilities[2]->targetEnemy = true;

	level = 10;


}

void Agent::Update(Player* players[], float msec)
{
	//run the relevant state
	states[subState](players, *this, msec);

	for (int i = 0; i < MAXABILITIES; ++i)
	{
		if (myAbilities[i]->cooldown > 0.0f)
		{
			myAbilities[i]->cooldown -= msec;
		}
	}
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