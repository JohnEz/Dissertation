#include "SphereNode.h"

SphereNode::SphereNode(void)	{
	target = NULL;
}

SphereNode::SphereNode(Quaternion orientation, Vector3 position, float r) {
	m_orientation	= orientation;
	m_position		= position;
	radius			= r;
}

SphereNode::~SphereNode(void)	{

}