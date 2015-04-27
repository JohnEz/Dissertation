#include "../../nclgl/Vector3.h"

enum CollisionVolumeType {COLLISION_VOL_SPHERE, COLLISION_VOL_PLANE, COLLISION_VOL_AABB, COLLISION_VOL_CYLINDER, COLLISION_VOL_LEAF };

class CollisionVolume {
public:
	virtual CollisionVolumeType GetType () const = 0;
};

class CollisionSphere : public CollisionVolume {
public :
	CollisionSphere(float radius) : radius(radius) {}

	CollisionVolumeType GetType () const {
		return COLLISION_VOL_SPHERE;
	}

	float GetRadius () const {
		return radius;
	}

private :
	float radius;
};

class CollisionLeaf : public CollisionVolume {
public :
	CollisionLeaf(float radius) : radius(radius) {}

	CollisionVolumeType GetType () const {
		return COLLISION_VOL_LEAF;
	}

	float GetRadius () const {
		return radius;
	}

private :
	float radius;
};

class CollisionCylinder : public CollisionVolume {
public :
	CollisionCylinder(float radius, Vector3 startPoint, Vector3 endPoint) : radius(radius), startPoint(startPoint), endPoint(endPoint) {}

	CollisionVolumeType GetType () const {
		return COLLISION_VOL_CYLINDER;
	}

	float GetRadius () const {
		return radius;
	}

	Vector3 GetStart () const {
		return startPoint;
	}

	Vector3 GetEnd () const {
		return endPoint;
	}

private :
	float radius;
	Vector3 startPoint;
	Vector3 endPoint;
};

class CollisionAABB : public CollisionVolume {
public:
	CollisionAABB(Vector3 halfDim) : halfDim(halfDim) {}

	CollisionVolumeType GetType () const {
		return COLLISION_VOL_AABB;
	}

	Vector3 getHalfDimensions () const { return halfDim; }

private:
	Vector3 halfDim;
};

class CollisionPlane : public CollisionVolume {
public:

	CollisionPlane(Vector3 normal, float distance): distance(distance), normal(normal) {}

	CollisionVolumeType GetType() const {
		return COLLISION_VOL_PLANE;
	}

	Vector3 GetNormal() const {
		return normal;
	}

	float GetDistance() const {
		return distance;
	}

private:
	float distance;
	Vector3 normal;
};

class CollisionData {
public:
	Vector3 m_point;
	Vector3 m_normal;
	float m_penetration;
};
