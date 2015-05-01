#include "GameClass.h"

GameClass* GameClass::instance = NULL;

GameClass::GameClass()	{
	renderCounter	= 0.0f;
	physicsCounter	= 0.0f;

	instance		= this;
}

GameClass::~GameClass(void)	{
	for(vector<GameEntity*>::iterator i = allEntities.begin(); i != allEntities.end(); ++i) {
		delete (*i);
	}
	delete gameCamera;
}

void GameClass::UpdatePhysics(float msec) {
	physicsCounter	+= msec;
	PhysicsSystem::GetPhysicsSystem().Update(msec);
	while(physicsCounter >= 0.0f) {
		physicsCounter -= PHYSICS_TIMESTEP;
		
	}
}

void GameClass::UpdateRendering(float msec) {
	renderCounter	-= msec;

	Renderer::GetRenderer().UpdateScene(msec);
	Renderer::GetRenderer().RenderScene();

	if(renderCounter <= 0.0f) {	//Update our rendering logic
		//Renderer::GetRenderer().UpdateScene(1000.0f / (float)RENDER_HZ);
		//Renderer::GetRenderer().RenderScene();
		renderCounter += (1000.0f / (float)RENDER_HZ);
	}
}