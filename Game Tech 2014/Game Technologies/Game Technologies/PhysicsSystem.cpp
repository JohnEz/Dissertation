#include "PhysicsSystem.h"

PhysicsSystem* PhysicsSystem::instance = 0;

PhysicsSystem::PhysicsSystem(void)	{
	float half= WORLDSIZE/2;
	float quater= WORLDSIZE/4;
	float third = WORLDSIZE/3;
	float sixth = WORLDSIZE/6;

	WorldPartition* p1 = new WorldPartition();
	WorldPartition* p2 = new WorldPartition();
	WorldPartition* p3 = new WorldPartition();
	WorldPartition* p4 = new WorldPartition();
	WorldPartition* p5 = new WorldPartition();
	WorldPartition* p6 = new WorldPartition();
	WorldPartition* p7 = new WorldPartition();
	WorldPartition* p8 = new WorldPartition();
	WorldPartition* p9 = new WorldPartition();
	WorldPartition* p10 = new WorldPartition();
	WorldPartition* p11 = new WorldPartition();
	WorldPartition* p12 = new WorldPartition();
	WorldPartition* p13 = new WorldPartition();
	WorldPartition* p14 = new WorldPartition();
	WorldPartition* p15 = new WorldPartition();
	WorldPartition* p16 = new WorldPartition();
	WorldPartition* p17 = new WorldPartition();
	WorldPartition* p18 = new WorldPartition();

	p1->pos = Vector3(-third * 2, half, -third * 2);
	p2->pos = Vector3(0			, half, -third * 2);
	p3->pos = Vector3(third * 2	, half, -third * 2);
	p4->pos = Vector3(-third * 2, half, 0);
	p5->pos = Vector3(0			, half, 0);
	p6->pos = Vector3(third * 2	, half, 0);
	p7->pos = Vector3(-third * 2, half, third * 2);
	p8->pos = Vector3(0			, half, third * 2);
	p9->pos = Vector3(third * 2	, half, third * 2);

	p10->pos = Vector3(-third * 2	, -half, -third * 2);
	p11->pos = Vector3(0			, -half, -third * 2);
	p12->pos = Vector3(third * 2	, -half, -third * 2);
	p13->pos = Vector3(-third * 2	, -half, 0);
	p14->pos = Vector3(0			, -half, 0);
	p15->pos = Vector3(third * 2	, -half, 0);
	p16->pos = Vector3(-third * 2	, -half, third * 2);
	p17->pos = Vector3(0			, -half, third * 2);
	p18->pos = Vector3(third * 2	, -half, third * 2);

	allPartitions.push_back(p1);
	allPartitions.push_back(p2);
	allPartitions.push_back(p3);
	allPartitions.push_back(p4);
	allPartitions.push_back(p5);
	allPartitions.push_back(p6);
	allPartitions.push_back(p7);
	allPartitions.push_back(p8);
	allPartitions.push_back(p9);
	allPartitions.push_back(p10);
	allPartitions.push_back(p11);
	allPartitions.push_back(p12);
	allPartitions.push_back(p13);
	allPartitions.push_back(p14);
	allPartitions.push_back(p15);
	allPartitions.push_back(p16);
	allPartitions.push_back(p17);
	allPartitions.push_back(p18);

	halfDim = Vector3(third, half, third);


}

PhysicsSystem::~PhysicsSystem(void)	{

}

void	PhysicsSystem::Update(float msec) {	
	BroadPhaseCollisions();
	NarrowPhaseCollisions();

	for(vector<PhysicsNode*>::iterator i = allNodes.begin(); i != allNodes.end(); ++i) {
		(*i)->Update(msec);
	}

	Performance::GetInstance()->calculatePPS(msec);
}

void	PhysicsSystem::BroadPhaseCollisions() {
	//create boundingboxs
	//3x3x2

	for (int i = 0; i < allPartitions.size(); i++) {
		allPartitions[i]->myNodes.clear();
		for (int j = 0; j < allNodes.size(); j++) {
			PhysicsNode& node = *allNodes[j];

			if (node.GetCollisionVolume()->GetType() == COLLISION_VOL_PLANE)
			{
				allPartitions[i]->myNodes.push_back(allNodes[j]);
			}
			else if (CheckBounding(node, allPartitions[i]->pos, halfDim))
			{
				allPartitions[i]->myNodes.push_back(allNodes[j]);
			}
		}
	}
}

bool PhysicsSystem::CheckBounding(PhysicsNode& n, Vector3 pos, Vector3 halfDim)
{
	CollisionSphere& sphere = *(CollisionSphere*)n.GetCollisionVolume();

	float dist = abs(pos.x - n.GetPosition().x);
	float sum = halfDim.x + sphere.GetRadius();

	if(dist <= sum) {
		dist = abs(pos.y - n.GetPosition().y);
		sum = halfDim.y + sphere.GetRadius();

		if(dist <= sum) {
			dist = abs(pos.z - n.GetPosition().z);
			sum = halfDim.z + sphere.GetRadius();

			if(dist <= sum) {
				//if there is collision data storage
				return true;
			}
		}
	}
	return false;
}

void	PhysicsSystem::NarrowPhaseCollisions() {
	//temp, for all nodes
	for (int k = 0; k < allPartitions.size(); k++) {
		for (int i = 0; i < allPartitions[k]->myNodes.size(); i++) {

			//get the first node
			PhysicsNode& first = *allPartitions[k]->myNodes[i];

			if (!allPartitions[k]->myNodes[i]) continue;

			CollisionVolume* fv = first.GetCollisionVolume();

			if (!fv) continue;

			// loop througgh the rest of the nodes
			for (int j = i + 1; j < allPartitions[k]->myNodes.size(); j++) {

				//get the other node
				PhysicsNode& second = *allPartitions[k]->myNodes[j];
				CollisionVolume* sv = second.GetCollisionVolume();
				if (!sv) continue;

				//CollisionData data;

				//find collision type
				switch(fv->GetType()) {
				case COLLISION_VOL_SPHERE:
				
					switch(sv->GetType()) {
					case COLLISION_VOL_SPHERE:
						//if they are both spheres
						CollisionData data;
						//have they colided?
						if (CollisionDetector::SphereSphereCollision(first, second, &data)) {
							//collision response
							CollisionDetector::AddCollisionImpulse(first, second, data);
							first.SetAtRest(false);
							second.SetAtRest(false);

							if (first.GetMaxCol() != -1)
							{
								first.AddCol(1);
							}

							if (second.GetMaxCol() != -1)
							{
								second.AddCol(1);
							}

							float score = abs(first.GetLinearVelocity().x) + abs(first.GetLinearVelocity().y) + abs(first.GetLinearVelocity().z);
							score += abs(second.GetLinearVelocity().x) + abs(second.GetLinearVelocity().y) + abs(second.GetLinearVelocity().z);
							score = score * 10;

							Performance::GetInstance()->addScore(score);
							Performance::GetInstance()->addCollisions();
						}
						continue;
					}

				case COLLISION_VOL_PLANE:
					switch(sv->GetType()) {
					case COLLISION_VOL_SPHERE:
						//if the first is a plane and second is a sphere
						CollisionData data;

						//have they colided?
						if (CollisionDetector::SpherePlaneCollision(second, first, &data)) {
							//collision response
							CollisionDetector::AddCollisionImpulse(second, first, data);
						}
						continue;
					}

				case COLLISION_VOL_CYLINDER:
					switch(sv->GetType()) {
					case COLLISION_VOL_SPHERE:
						//if the first is a cylinder and second is a sphere
						CollisionData data;

						//have they colided?
						if (CollisionDetector::CylinderSphereCollision(first, second, &data)) {
							//collision response
							CollisionDetector::AddCollisionImpulse(first, second, data);

							float score = abs(first.GetLinearVelocity().x) + abs(first.GetLinearVelocity().y) + abs(first.GetLinearVelocity().z);
							score += abs(second.GetLinearVelocity().x) + abs(second.GetLinearVelocity().y) + abs(second.GetLinearVelocity().z);
							score = score * 10;

							Performance::GetInstance()->addScore(score);

							Performance::GetInstance()->addCollisions();
						}
						continue;
					}

					case COLLISION_VOL_LEAF:
					switch(sv->GetType()) {
						case COLLISION_VOL_SPHERE:
						//if they are both spheres
						CollisionData data;
						//have they colided?
						if (CollisionDetector::SphereSphereCollision(first, second, &data)) {
							//collision response
							CollisionDetector::AddCollisionImpulse(first, second, data);
							second.SetAtRest(false);

							float score = abs(first.GetLinearVelocity().x) + abs(first.GetLinearVelocity().y) + abs(first.GetLinearVelocity().z);
							score += abs(second.GetLinearVelocity().x) + abs(second.GetLinearVelocity().y) + abs(second.GetLinearVelocity().z);
							score = score * 10;

							Performance::GetInstance()->addScore(score);

							Performance::GetInstance()->addCollisions();
						}
						continue;
				
					}
				}
		
			}
		}
	}
}

void	PhysicsSystem::NarrowPhaseCollisionsOLD() {
	//temp, for all nodes
	for (int i = 0; i < allNodes.size(); i++) {

		//get the first node
		PhysicsNode& first = *allNodes[i];

		if (!allNodes[i]) continue;

		CollisionVolume* fv = first.GetCollisionVolume();

		if (!fv) continue;

		// loop througgh the rest of the nodes
		for (int j = i + 1; j < allNodes.size(); j++) {

			//get the other node
			PhysicsNode& second = *allNodes[j];
			CollisionVolume* sv = second.GetCollisionVolume();
			if (!sv) continue;

			//CollisionData data;

			//find collision type
			switch(fv->GetType()) {
			case COLLISION_VOL_SPHERE:
				
				switch(sv->GetType()) {
				case COLLISION_VOL_SPHERE:
					//if they are both spheres
					CollisionData data;
					//have they colided?
					if (CollisionDetector::SphereSphereCollision(first, second, &data)) {
						//collision response
						CollisionDetector::AddCollisionImpulse(first, second, data);
						first.SetAtRest(false);
						second.SetAtRest(false);

						if (first.GetMaxCol() != -1)
						{
							first.AddCol(1);
						}

						if (second.GetMaxCol() != -1)
						{
							second.AddCol(1);
						}


						Performance::GetInstance()->addCollisions();
					}
					continue;
				}

			case COLLISION_VOL_PLANE:
				switch(sv->GetType()) {
				case COLLISION_VOL_SPHERE:
					//if the first is a plane and second is a sphere
					CollisionData data;

					//have they colided?
					if (CollisionDetector::SpherePlaneCollision(second, first, &data)) {
						//collision response
						CollisionDetector::AddCollisionImpulse(second, first, data);
					}
					continue;
				}

			case COLLISION_VOL_CYLINDER:
				switch(sv->GetType()) {
				case COLLISION_VOL_SPHERE:
					//if the first is a cylinder and second is a sphere
					CollisionData data;

					//have they colided?
					if (CollisionDetector::CylinderSphereCollision(first, second, &data)) {
						//collision response
						CollisionDetector::AddCollisionImpulse(first, second, data);

						//Vector3 vel = second.GetLinearVelocity();
						//second.SetLinearVelocity(-vel);

						Performance::GetInstance()->addCollisions();
					}
					continue;
				}

				case COLLISION_VOL_LEAF:
				switch(sv->GetType()) {
					case COLLISION_VOL_SPHERE:
					//if they are both spheres
					CollisionData data;
					//have they colided?
					if (CollisionDetector::SphereSphereCollision(first, second, &data)) {
						//collision response
						CollisionDetector::AddCollisionImpulse(first, second, data);
						second.SetAtRest(false);

						Performance::GetInstance()->addCollisions();
					}
					continue;
				
				}
			}
		
		}
	}
}

void	PhysicsSystem::AddNode(PhysicsNode* n) {
	allNodes.push_back(n);
}

void	PhysicsSystem::RemoveNode(PhysicsNode* n) {
	for(vector<PhysicsNode*>::iterator i = allNodes.begin(); i != allNodes.end(); ++i) {
		if((*i) == n) {
			allNodes.erase(i);
			return;
		}
	}
}