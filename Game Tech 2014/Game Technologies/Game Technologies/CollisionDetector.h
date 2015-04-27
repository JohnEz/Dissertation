#include "PhysicsNode.h"

class CollisionDetector {
public:
	
	static bool SphereSphereCollision(PhysicsNode& p0, PhysicsNode& p1, CollisionData* data = NULL);

	static bool SpherePlaneCollision(PhysicsNode& p0, PhysicsNode& p1, CollisionData* data = NULL);

	static void AddCollisionImpulse(PhysicsNode& p0, PhysicsNode& p1, CollisionData& data);

	static bool AABBCollision(PhysicsNode& p0, PhysicsNode& p1, CollisionData* data = NULL);

	static bool AABBSphereCollision(PhysicsNode& p0, PhysicsNode& p1, CollisionData* data = NULL);

	static bool CylinderSphereCollision(PhysicsNode& p0, PhysicsNode& p1, CollisionData* data = NULL);
};

inline float LengthSq(Vector3 v) {
	return Vector3::Dot(v, v);
}