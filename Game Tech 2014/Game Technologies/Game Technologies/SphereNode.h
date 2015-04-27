#include "PhysicsNode.h"

class SphereNode : PhysicsNode
{
public:
	SphereNode(void);
	SphereNode(Quaternion orientation, Vector3 position, float r);
	~SphereNode(void);

protected:
	float radius;
};