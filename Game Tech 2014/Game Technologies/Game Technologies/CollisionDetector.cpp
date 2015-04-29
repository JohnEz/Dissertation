#include "CollisionDetector.h"

bool CollisionDetector::SphereSphereCollision(PhysicsNode& p0, PhysicsNode& p1, CollisionData* data) {
	
	return false;

	//get the collision data
	CollisionSphere& s0 = *(CollisionSphere*)p0.GetCollisionVolume();
	CollisionSphere& s1 = *(CollisionSphere*)p1.GetCollisionVolume();

	//get the normal
	Vector3 normal = p0.GetPosition() - p1.GetPosition();

	//get the distance (squared)
	const float distSq = LengthSq(normal);

	//get the max distance before collision
	const float sumRadius = s0.GetRadius() + s1.GetRadius();

	//if the distance is less than the max distance
	if (distSq < sumRadius*sumRadius) {
		//if there is collision data storage
		if (data) {
			//set the penetration depth
			data->m_penetration = sumRadius - sqrtf(distSq);
			//get the normal of the collision
			normal.Normalise();
			data->m_normal = normal;
			//get the point of collision
			data->m_point = p0.GetPosition() - normal*(s0.GetRadius() - data->m_penetration*0.5f);
		}
		return true;
	}
	return false;
}

bool CollisionDetector::SpherePlaneCollision(PhysicsNode& p0, PhysicsNode& p1, CollisionData* data) {

	//get collision data
	CollisionSphere& sphere = *(CollisionSphere*)p0.GetCollisionVolume();
	CollisionPlane& plane = *(CollisionPlane*)p1.GetCollisionVolume();
	
	
	//get separation
	float separation = Vector3::Dot(p0.GetPosition(), plane.GetNormal()) - plane.GetDistance();

	// if the separation is greater than the radius it cant of collided
	if (separation > sphere.GetRadius()) {
		return false;
	}

	//if there is collision data storage
	if (data) {
		//set the penetration depth
		data->m_penetration = sphere.GetRadius() - separation;
		//get the normal of the collision
		data->m_normal = plane.GetNormal();
		//get the point of collision
		data->m_point = p0.GetPosition() - plane.GetNormal()*separation;
	}

	return true;
}

void CollisionDetector::AddCollisionImpulse(PhysicsNode &p0, PhysicsNode &p1, CollisionData &data) {

	// if neither objects have mass
	if(p0.GetInverseMass() + p1.GetInverseMass() == 0.0f) {
		return;
	}

	Vector3 r0 = data.m_point - p0.GetPosition();
	Vector3 r1 = data.m_point - p1.GetPosition();

	//get velocities
	Vector3 v0 = p0.GetLinearVelocity() + Vector3::Cross(p0.GetAngularVelocity(), r0);
	Vector3 v1 = p1.GetLinearVelocity() + Vector3::Cross(p1.GetAngularVelocity(), r1);
	Vector3 dv = v0 - v1;

	float relMov = -Vector3::Dot(dv, data.m_normal);

	//if(relMov < -0.01f) return;
	
	{
		float e = 0.0f;
		float normDiv = (p0.GetInverseMass() + p1.GetInverseMass()) + Vector3::Dot(data.m_normal, Vector3::Cross(p0.GetInverseInertia()
		* Vector3::Cross(r0, data.m_normal), r0) + Vector3::Cross(p1.GetInverseInertia() * Vector3::Cross(r1, data.m_normal), r1));
		float jn = -1 * (1+e) * Vector3::Dot(dv, data.m_normal) / normDiv;

		jn = jn + (data.m_penetration * 0.1f);

		//set p0's velocity
		Vector3 l0 = p0.GetLinearVelocity() + data.m_normal * (jn * p0.GetInverseMass());
		p0.SetLinearVelocity(l0);

		//set p0's rotation velocity
		Vector3 a0 = p0.GetAngularVelocity() + p0.GetInverseInertia() * Vector3::Cross(r0, data.m_normal * jn);
		p0.SetAngularVelocity(a0);

		//set p1's velocity
		Vector3 l1 = p1.GetLinearVelocity() - data.m_normal * ( jn * p1.GetInverseMass());
		p1.SetLinearVelocity(l1);

		//set p1's rotation velocity
		Vector3 a1 = p1 . GetAngularVelocity () - p1.GetInverseInertia() * Vector3::Cross(r1, data.m_normal * jn);
		p1.SetAngularVelocity(a1);
	}

	//if (false)
	{
		Vector3 tangent = dv - data.m_normal * Vector3::Dot(dv, data.m_normal);
		tangent.Normalise();
		float tangDiv = (p0.GetInverseMass() + p1.GetInverseMass()) + Vector3::Dot(tangent, Vector3::Cross(p0.GetInverseInertia() * Vector3::Cross(r0,
		tangent),r0) + Vector3::Cross(p1.GetInverseInertia() * Vector3::Cross(r1, tangent), r1));

		float jt = -1 * Vector3::Dot(dv, tangent) / tangDiv;

		Vector3 l0 = p0.GetLinearVelocity() + tangent * (jt * p0.GetInverseMass());
		//p0.SetLinearVelocity(l0);

		Vector3 a0 = p0.GetAngularVelocity() + p0.GetInverseInertia() * Vector3::Cross(r0, tangent * jt);
		p0.SetAngularVelocity(a0);

		Vector3 l1 = p1.GetLinearVelocity() - tangent * (jt * p1.GetInverseMass());
		//p1.SetLinearVelocity(l1);

		Vector3 a1 = p1.GetAngularVelocity() - p1.GetInverseInertia() * Vector3::Cross(r1, tangent * jt);
		p1.SetAngularVelocity(a1);
	}
}

bool CollisionDetector::AABBCollision(PhysicsNode& p0, PhysicsNode& p1, CollisionData* data) {
	CollisionAABB& aabb0 = *(CollisionAABB*)p0.GetCollisionVolume();
	CollisionAABB& aabb1 = *(CollisionAABB*)p1.GetCollisionVolume();

	float dist = abs(p0.GetPosition().x - p1.GetPosition().x);
	float sum = aabb0.getHalfDimensions().x + aabb1.getHalfDimensions().x;

	if(dist <= sum) {
		dist = abs(p0.GetPosition().y - p1.GetPosition().y);
		sum = aabb0.getHalfDimensions().y + aabb1.getHalfDimensions().y;

		if(dist <= sum) {
			dist = abs(p0.GetPosition().z - p1.GetPosition().z);
			sum = aabb0.getHalfDimensions().z + aabb1.getHalfDimensions().z;

			if(dist <= sum) {
				return true;
			}
		}
	}
	return false;
}

bool CollisionDetector::AABBSphereCollision(PhysicsNode& p0, PhysicsNode& p1, CollisionData* data) {
	CollisionAABB& aabb0 = *(CollisionAABB*)p0.GetCollisionVolume();
	CollisionSphere& sphere = *(CollisionSphere*)p1.GetCollisionVolume();

	float dist = abs(p0.GetPosition().x - p1.GetPosition().x);
	float sum = aabb0.getHalfDimensions().x + sphere.GetRadius();

	if(dist <= sum) {
		dist = abs(p0.GetPosition().y - p1.GetPosition().y);
		sum = aabb0.getHalfDimensions().y + sphere.GetRadius();

		if(dist <= sum) {
			dist = abs(p0.GetPosition().z - p1.GetPosition().z);
			sum = aabb0.getHalfDimensions().z + sphere.GetRadius();

			if(dist <= sum) {
				//if there is collision data storage
				return true;
			}
		}
	}
	return false;
}

bool CollisionDetector::CylinderSphereCollision(PhysicsNode& p0, PhysicsNode& p1, CollisionData* data) {
	CollisionCylinder& cylinder = *(CollisionCylinder*)p0.GetCollisionVolume();
	CollisionSphere& sphere = *(CollisionSphere*)p1.GetCollisionVolume();

	Vector3 cylCenterVector = cylinder.GetEnd() - cylinder.GetStart();

	Vector3 pos1 = p1.GetPosition() - cylinder.GetStart();

	float distanceFactorFromEP1 = Vector3::Dot(p1.GetPosition() - cylinder.GetStart(), cylCenterVector) / Vector3::Dot(cylCenterVector, cylCenterVector);
	if(distanceFactorFromEP1 < 0) distanceFactorFromEP1 = 0;// clamp to endpoints if neccesary
	if(distanceFactorFromEP1 > 1) distanceFactorFromEP1 = 1;
	Vector3 closestPoint = cylinder.GetStart() + (cylCenterVector * distanceFactorFromEP1);

	Vector3 collisionVector = p1.GetPosition() - closestPoint;
	float distance = collisionVector.Length();
	Vector3 collisionNormal = collisionVector / distance;

	if(distance < sphere.GetRadius() + cylinder.GetRadius())
	{
	  //collision occurred. use collisionNormal to reflect sphere off cyl

		float factor = Vector3::Dot(p1.GetLinearVelocity(), collisionNormal);

		p1.SetLinearVelocity(p1.GetLinearVelocity() - (collisionNormal * factor * 0.8f));

		const float distSq = LengthSq(collisionNormal);

		//get the max distance before collision
		const float sumRadius = sphere.GetRadius() + cylinder.GetRadius();

		p1.SetPosition(p1.GetPosition() + Vector3(collisionNormal * (sumRadius - sqrtf(distSq))));
		return true;
	}
	return false;
}