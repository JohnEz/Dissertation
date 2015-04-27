#include "MyGame.h"
#include "Agent.h"

/*
Creates a really simple scene for our game - A cube robot standing on
a floor. As the module progresses you'll see how to get the robot moving
around in a physically accurate manner, and how to stop it falling
through the floor as gravity is added to the scene. 

You can completely change all of this if you want, it's your game!

*/

MyGame::MyGame()	{
	gameCamera = new Camera(-30.0f,0.0f,Vector3(0,450,850));

	Renderer::GetRenderer().SetCamera(gameCamera);

	oldState = false;
	currentState = false;

	sphereSize = 10.0f;
	speed = 1.0f;

	/*
	We're going to manage the meshes we need in our game in the game class!

	You can do this with textures, too if you want - but you might want to make
	some sort of 'get / load texture' part of the Renderer or OGLRenderer, so as
	to encapsulate the API-specific parts of texture loading behind a class so
	we don't care whether the renderer is OpenGL / Direct3D / using SOIL or 
	something else...
	*/
	cube	= new OBJMesh(MESHDIR"cube.obj");
	quad	= Mesh::GenerateQuad();
	sphere	= new OBJMesh(MESHDIR"ico.obj");

	//floor
	allEntities.push_back(BuildQuadEntity(10000.0f));

	//allEntities.push_back(BuildFloatSphereEntity(100.0f, Vector3(200, 300, 156)));

	//allEntities.push_back(BuildStemEntity(Vector3(0, 250, 0)));
	allAgents.push_back(AddAgent(Vector3(0, 250, 0)));
	//allAgents.push_back(AddAgent(Vector3(100, 250, 100)));
	//allAgents.push_back(AddAgent(Vector3(200, 250, 400)));
	//allAgents.push_back(AddAgent(Vector3(300, 250, 300)));
	//allAgents.push_back(AddAgent(Vector3(400, 250, 200)));


}

MyGame::~MyGame(void)	{
	/*
	We're done with our assets now, so we can delete them
	*/
	delete cube;
	delete quad;
	delete sphere;

	CubeRobot::DeleteCube();

	//GameClass destructor will destroy your entities for you...
}

/*
Here's the base 'skeleton' of your game update loop! You will presumably
want your games to have some sort of internal logic to them, and this
logic will be added to this function.
*/
void MyGame::UpdateGame(float msec) {
	if(gameCamera) {
		gameCamera->UpdateCamera(msec);
	}

	for(vector<GameEntity*>::iterator i = allEntities.begin(); i != allEntities.end(); ++i) {
		(*i)->Update(msec);
	}

	//update all the agents
	for(vector<Agent*>::iterator i = allAgents.begin(); i != allAgents.end(); ++i) {
		(*i)->Update(msec);
	}

	oldState = currentState;
	currentState = Window::GetMouse()->ButtonDown(MOUSE_LEFT);

	if (!oldState && currentState)
	{
		float pitch = gameCamera->GetPitch();
		float yaw = gameCamera->GetYaw();

		pitch = pitch * PI / 180.0f;
		yaw = yaw * PI / 180.0f;

		Vector3 norm = Vector3(0, 0, 0);

		norm.x = -cos(pitch) * sin(yaw);
		norm.y = sin(pitch);
		norm.z = -cos(pitch) * cos(yaw);

		norm.Normalise();

		allEntities.push_back(BuildSphereEntity(sphereSize, gameCamera->GetPosition(), norm * speed));
	}

	if (Window::GetKeyboard()->KeyDown(KEYBOARD_E))
	{
		sphereSize += 0.05f * msec;
		if (sphereSize > 100.0f)
		{
			sphereSize = 100.0f;
		}
	}
	else if (Window::GetKeyboard()->KeyDown(KEYBOARD_Q))
	{
		sphereSize -= 0.05f * msec;
		if (sphereSize < 10.0f)
		{
			sphereSize = 10.0f;
		}
	}

	if (Window::GetMouse()->ButtonDown(MOUSE_RIGHT))
	{
		speed += 0.0005f * msec;
		if (speed > 3.0f)
		{
			speed = 3.0f;
		}
	}
	else if (Window::GetMouse()->ButtonDown(MOUSE_MIDDLE))
	{
		speed -= 0.0005f * msec;
		if (speed < 0.5f)
		{
			speed = 0.5f;
		}
	}

	/*
	Here's how we can use OGLRenderer's inbuilt debug-drawing functions! 
	I meant to talk about these in the graphics module - Oops!

	We can draw squares, lines, crosses and circles, of varying size and
	colour - in either perspective or orthographic mode.

	Orthographic debug drawing uses a 'virtual canvas' of 720 * 480 - 
	that is 0,0 is the top left, and 720,480 is the bottom right. A function
	inside OGLRenderer is provided to convert clip space coordinates into
	this canvas space coordinates. How you determine clip space is up to you -
	maybe your renderer has getters for the view and projection matrix?

	Or maybe your Camera class could be extended to contain a projection matrix?
	Then your game would be able to determine clip space coordinates for its
	active Camera without having to involve the Renderer at all?

	Perspective debug drawing relies on the view and projection matrices inside
	the renderer being correct at the point where 'SwapBuffers' is called. As
	long as these are valid, your perspective drawing will appear in the world.

	This gets a bit more tricky with advanced rendering techniques like deferred
	rendering, as there's no guarantee of the state of the depth buffer, or that
	the perspective matrix isn't orthographic. Therefore, you might want to draw
	your debug lines before the inbuilt position before SwapBuffers - there are
	two OGLRenderer functions DrawDebugPerspective and DrawDebugOrtho that can
	be called at the appropriate place in the pipeline. Both take in a viewProj
	matrix as an optional parameter.

	Debug rendering uses its own debug shader, and so should be unaffected by
	and shader changes made 'outside' of debug drawing

	*/
	//Lets draw a box around the cube robot!
	Renderer::GetRenderer().DrawDebugBox(DEBUGDRAW_PERSPECTIVE, Vector3(0,51,0), Vector3(100,100,100), Vector3(1,0,0));

	////We'll assume he's aiming at something...so let's draw a line from the cube robot to the target
	////The 1 on the y axis is simply to prevent z-fighting!
	Renderer::GetRenderer().DrawDebugLine(DEBUGDRAW_PERSPECTIVE, Vector3(0,1,0),Vector3(200,1,200), Vector3(0,0,1), Vector3(1,0,0));

	////Maybe he's looking for treasure? X marks the spot!
	Renderer::GetRenderer().DrawDebugCross(DEBUGDRAW_PERSPECTIVE, Vector3(200,1,200),Vector3(50,50,50), Vector3(0,0,0));

	////CubeRobot is looking at his treasure map upside down!, the treasure's really here...
	Renderer::GetRenderer().DrawDebugCircle(DEBUGDRAW_PERSPECTIVE, Vector3(-200,1,-200),50.0f, Vector3(0,1,0));
}

/*
Makes an entity that looks like a CubeRobot! You'll probably want to modify
this so that you can have different sized robots, with different masses and
so on!
*/
GameEntity* MyGame::BuildRobotEntity() {
	GameEntity*g = new GameEntity(new CubeRobot(), new PhysicsNode());

	g->ConnectToSystems();
	return g;
}

/*
Makes a cube. Every game has a crate in it somewhere!
*/
GameEntity* MyGame::BuildCubeEntity(float size, Vector3 pos) {
	SceneNode* c = new SceneNode(cube);
	
	c->SetModelScale(Vector3(size,size,size));
	c->SetBoundingRadius(size);
	c->SetColour(Vector4(1,1,1,1));

	PhysicsNode* p = new PhysicsNode();
	p->SetPosition(pos);
	p->SetUseGravity(false);
	p->SetLinearVelocity(Vector3(0, 0, 0));
	p->SetAngularVelocity(Vector3(0, 0, 0));
	
	float Ixx = 12.0f/(1*(size*size + size*size));
	float Iyy = 12.0f/(1*(size*size + size*size));
	float Izz = 12.0f/(1*(size*size + size*size));
	//float elements[] = {Ixx, 0, 0, 0, 0, Iyy, 0, 0, 0, 0, Izz, 0, 0, 0, 0, 1};

	float elements[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};

	Matrix4 mat = Matrix4(elements);

	p->SetInverseInertia(mat);
	p->SetInverseMass(1.0f);
	p->SetCollisionVolume(new CollisionAABB(Vector3(size, size, size)));

	GameEntity*g = new GameEntity(c, p);
	g->ConnectToSystems();
	return g;
}

/*
Makes a sphere.
*/
GameEntity* MyGame::BuildSphereEntity(float radius, Vector3 pos, Vector3 vel) {
	SceneNode* s = new SceneNode(sphere);

	s->SetModelScale(Vector3(radius,radius,radius));
	s->SetBoundingRadius(radius);
	s->SetColour(Vector4(0,0,1,1));

	PhysicsNode*p = new PhysicsNode();
	p->SetPosition(pos);
	p->SetLinearVelocity(vel);
	p->SetAngularVelocity(Vector3(0, 0, 0));

	float I = 2.5f/(1.0f*radius*radius);
	float elements[] = {I, 0, 0, 0, 0, I, 0, 0, 0, 0, I, 0, 0, 0, 0, 1};
	Matrix4 mat = Matrix4(elements);
	p->SetInverseInertia(mat);

	p->SetInverseMass(1.0f);

	p->SetCollisionVolume(new CollisionSphere(radius));

	GameEntity*g = new GameEntity(s, p);
	g->ConnectToSystems();
	return g;
}

/*
Makes a floating sphere.
*/
GameEntity* MyGame::BuildFloatSphereEntity(float radius, Vector3 pos) {
	SceneNode* s = new SceneNode(sphere);

	s->SetModelScale(Vector3(radius,radius,radius));
	s->SetBoundingRadius(radius);
	s->SetColour(Vector4(0,0,1,1));

	PhysicsNode*p = new PhysicsNode();
	p->SetPosition(pos);
	p->SetLinearVelocity(Vector3(0.1f, 0, 0));
	p->SetAngularVelocity(Vector3(0, 0, 0));
	p->SetMaxCol(5);

	float I = 2.5f/(1.0f*radius*radius);
	float elements[] = {I, 0, 0, 0, 0, I, 0, 0, 0, 0, I, 0, 0, 0, 0, 1};
	Matrix4 mat = Matrix4(elements);
	p->SetInverseInertia(mat);

	p->SetInverseMass(1.0f);

	p->SetCollisionVolume(new CollisionSphere(radius));

	GameEntity*g = new GameEntity(s, p);
	g->ConnectToSystems();

	g->GetPhysicsNode().SetAtRest(true);
	return g;
}

/*
Makes a flat quad, initially oriented such that we can use it as a simple
floor. 
*/
GameEntity* MyGame::BuildQuadEntity(float size) {
	SceneNode* s = new SceneNode(quad);

	s->SetModelScale(Vector3(size,size,size));
	s->SetBoundingRadius(size);

	PhysicsNode*p = new PhysicsNode(Quaternion::AxisAngleToQuaterion(Vector3(1,0,0), 90.0f), Vector3());
	p->SetUseGravity(false);
	p->SetInverseMass(0.0f);

	float elements[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
	Matrix4 mat = Matrix4(elements);

	p->SetInverseInertia(mat);

	p->SetCollisionVolume(new CollisionPlane(Vector3(0,1,0), 0));
	GameEntity*g = new GameEntity(s, p);
	g->ConnectToSystems();
	return g;
}

/*
Makes a leaf.
*/
GameEntity* MyGame::BuildLeafEntity(float radius) {
	SceneNode* s = new SceneNode(sphere);

	s->SetModelScale(Vector3(radius,radius,radius));
	s->SetBoundingRadius(radius);
	s->SetColour(Vector4(0,0,1,1));

	PhysicsNode*p = new PhysicsNode();

	float elements[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
	Matrix4 mat = Matrix4(elements);
	p->SetInverseInertia(mat);

	p->SetInverseMass(0.0f);

	p->SetCollisionVolume(new CollisionSphere(radius));

	GameEntity*g = new GameEntity(s, p);
	g->ConnectToSystems();
	return g;
}

GameEntity* MyGame::BuildStemEntity(Vector3 pos) {
	PhysicsNode*p = new PhysicsNode();

	float elements[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
	Matrix4 mat = Matrix4(elements);
	p->SetInverseInertia(mat);
	p->SetUseGravity(false);
	p->SetPosition(pos);
	p->SetInverseMass(0.0f);

	Vector3 endPos = pos;
	endPos.y += 200;

	p->SetCollisionVolume(new CollisionCylinder(1.5f, pos, endPos));

	SceneNode* s = new Stem(p);

	s->SetModelScale(Vector3(10,10,10));
	s->SetBoundingRadius(1);
	s->SetColour(Vector4(0,1,1,1));

	GameEntity*g = new GameEntity(s, p);
	g->ConnectToSystems();
	return g;
}

Agent* MyGame::AddAgent(const Vector3 pos)
{
	PhysicsNode*p = new PhysicsNode();

	p->SetPosition(pos);
	p->SetLinearVelocity(Vector3(0, 0, 0));
	p->SetAngularVelocity(Vector3(0, 0, 0));

	float I = 2.5f/(1.0f*25.0f*25.0f);
	float elements[] = {I, 0, 0, 0, 0, I, 0, 0, 0, 0, I, 0, 0, 0, 0, 1};
	Matrix4 mat = Matrix4(elements);
	p->SetInverseInertia(mat);

	p->SetInverseMass(1.0f);

	p->SetCollisionVolume(new CollisionSphere(25.0f));

	SceneNode* s = new SceneNode(sphere);

	s->SetModelScale(Vector3(25.0f,25.0f,25.0f));
	s->SetBoundingRadius(25.0f);
	s->SetColour(Vector4(0,0,1,1));

	Agent* a = new Agent(s, p);
	a->ConnectToSystems();
	return a;
}