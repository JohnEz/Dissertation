#include "PhysicsNode.h"

const Vector3 PhysicsNode::gravity = Vector3(0,-0.001,0);

PhysicsNode::PhysicsNode(void) : vol(0)	{
	target = NULL;
	useGravity = true;
	atRest = false;
	maxCollisions = -1;
	currentCollisions = 0;
}

PhysicsNode::PhysicsNode(Quaternion orientation, Vector3 position) : vol(0) {
	m_orientation	= orientation;
	m_position		= position;
	useGravity		= true;
	atRest			= false;
	maxCollisions = -1;
	currentCollisions = 0;
}

PhysicsNode::~PhysicsNode(void)	{

}

//You will perform your per-object physics integration, here!
//I've added in a bit that will set the transform of the
//graphical representation of this object, too.
void	PhysicsNode::Update(float msec) {
	//FUN GOES HERE

	if (m_position.x < -WORLDSIZE)
	{
		m_position.x = -WORLDSIZE + 1.0f;
		m_linearVelocity.x = -m_linearVelocity.x;
	}
	else if (m_position.x > WORLDSIZE)
	{
		m_position.x = WORLDSIZE - 1.0f;
		m_linearVelocity.x = -m_linearVelocity.x;
	}

	if (m_position.y < -WORLDSIZE)
	{
		m_position.y = -WORLDSIZE + 1.0f;
		m_linearVelocity.y = -m_linearVelocity.y;
	}
	else if (m_position.y > WORLDSIZE)
	{
		m_position.y = WORLDSIZE - 1.0f;
		m_linearVelocity.y = -m_linearVelocity.y;
	}

	if (m_position.z < -WORLDSIZE)
	{
		m_position.z = -WORLDSIZE + 1.0f;
		m_linearVelocity.z = -m_linearVelocity.z;
	}
	else if (m_position.z > WORLDSIZE)
	{
		m_position.z = WORLDSIZE - 1.0f;
		m_linearVelocity.z = -m_linearVelocity.z;
	}



	if (!atRest)
	{
		Vector3 acc = m_force * m_invMass + (useGravity? gravity : Vector3(0,0,0));

		symplecticEuler(acc, msec);

		calculateRotation(msec);

		if (m_linearVelocity.Length() < LINEAR_VELOCITY_MIN && m_angularVelocity.Length() < LINEAR_VELOCITY_MIN)
		{
			atRest = true;
		}
	}

	if(target) {
		target->SetTransform(BuildTransform());
	}

	
}

/*
This function simply turns the orientation and position
of our physics node into a transformation matrix, suitable
for plugging into our Renderer!

It is cleaner to work with matrices when it comes to rendering,
as it is what shaders expect, and allow us to keep all of our
transforms together in a single construct. But when it comes to
physics processing and 'game-side' logic, it is much neater to
have seperate orientations and positions.

*/
Matrix4		PhysicsNode::BuildTransform() {
	Matrix4 m = m_orientation.ToMatrix();

	m.SetPositionVector(m_position);

	return m;
}

void PhysicsNode::explicitEuler(Vector3 acc, float dT)
{
	m_position = m_position + (m_linearVelocity * dT);
	m_linearVelocity = m_linearVelocity + (acc * dT);
	m_linearVelocity = m_linearVelocity*LINEAR_VELOCITY_DAMP;
}

void PhysicsNode::implicitEuler(Vector3 acc, float dT)
{
	Vector3 v = m_linearVelocity + (acc * dT);
	m_position = m_position + (v * dT);
	m_linearVelocity = v;
	m_linearVelocity = m_linearVelocity*LINEAR_VELOCITY_DAMP;
}

void PhysicsNode::symplecticEuler(Vector3 acc, float dT)
{
	m_linearVelocity = m_linearVelocity + (acc * dT);
	m_linearVelocity = m_linearVelocity*LINEAR_VELOCITY_DAMP;
	m_position = m_position + (m_linearVelocity * dT);
}

//void PhysicsNode::verlet(Vector3& dis0, Vector3& dis1, Vector3& acc, float dT)
//{
//	Vector3 d = dis0 + (dis0 - dis1) + (acc * (dT * dT));
//	dis1 = dis0;
//	dis0 = d;
//}

void PhysicsNode::calculateRotation(float dT)
{
	//calculate the angular acc via inverse inertia * toque
	Vector3 angAcc = m_invInertia * m_torque;

	//calculate angular velocity usingg symplectic euler
	m_angularVelocity = m_angularVelocity + angAcc * dT;
	m_angularVelocity = m_angularVelocity*ANGULAR_VELOCITY_DAMP;
	m_orientation = m_orientation + m_orientation*(m_angularVelocity * (dT / 2.0f));

	//normalise the orientation
	m_orientation.Normalise();
}