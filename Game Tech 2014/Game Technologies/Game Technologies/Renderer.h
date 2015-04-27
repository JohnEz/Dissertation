/******************************************************************************
Class:Renderer		
Implements:OGLRenderer
Author:Rich Davison	<richard.davison4@newcastle.ac.uk> and YOU!
Description: For this module, you are being provided with a basic working
Renderer - to give you more time to work on your physics and AI!

It is basically the Renderer from the Graphics For Games Module as it was
by Tutorial 7 - Scene Management. It will split nodes up into those that are
opaque and transparent, and render accordingly. 

The only new bits are the ability to add and remove scene nodes, and set the
camera - as these are now to be controlled by your 'game'.

The renderer is one of those things that really there should be only one of.
I think it was a mistake to add support for multiple Windows / Renderers to
the code download, as it just overcomplicated things. To this end, The 
Renderer for this module is a singleton-like class, as with the 
PhysicsSystem. Thus, it is created, accessed, and destroyed via static 
functions. You don't have to do it this way if you don't want, and you are
of course free to add in the advanced graphical features from other tutorials
into this Renderer as you see fit. 


-_-_-_-_-_-_-_,------,   
_-_-_-_-_-_-_-|   /\_/\   NYANYANYAN
-_-_-_-_-_-_-~|__( ^ .^) /
_-_-_-_-_-_-_-""  ""   

*//////////////////////////////////////////////////////////////////////////////


#pragma once

#include "../../nclgl/OGLRenderer.h"
#include "../../nclgl/SceneNode.h"
#include "../../nclgl/Camera.h"
#include "../../nclgl/Frustum.h"
#include "../../nclgl/OBJmesh.h"
#include "../../nclgl/HeightMap.h"
#include "../../nclgl/text2D.h"
#include "Performance.h"
#include "LightStorage.h"
#include "TextMesh.h"
#include <algorithm>

#define LIGHTNUM 8

class Renderer : public OGLRenderer	{
public:
	virtual void RenderScene();
	virtual void UpdateScene(float msec);

	void	SetCamera(Camera*c);

	void	AddNode(SceneNode* n);

	void	RemoveNode(SceneNode* n);

	//Statics
	static bool Initialise() {
		instance = new Renderer(Window::GetWindow());
		return instance->HasInitialised();
	}

	static void Destroy() {
		delete instance;
	}
	
	static Renderer&GetRenderer() { return *instance;}

	void DrawSkybox();

	void	DrawText(const std::string &text, const Vector3 &position, const float size = 10.0f, const bool perspective = false);

protected:	
	Renderer(Window &parent);
	virtual ~Renderer(void);

	Mesh* quad;
	Mesh* quadW;
	GLuint cubeMap;
	OBJMesh* sphere; // Light volume

	void FillBuffers(); //G- Buffer Fill Render Pass
	void DrawPointLights(); // Lighting Render Pass
	void CombineBuffers(); // Combination Render Pass
	void GenerateScreenTexture(GLuint & into, bool depth = false); // Make a new texture...

	GLuint bufferFBO; // FBO for our G- Buffer pass
	GLuint bufferColourTex; // Albedo goes here
	GLuint bufferNormalTex; // Normals go here
	GLuint bufferDepthTex; // Depth goes here

	GLuint pointLightFBO; // FBO for our lighting pass
	GLuint lightEmissiveTex; // Store emissive lighting
	GLuint lightSpecularTex; // Store specular lighting

	void	BuildNodeLists(SceneNode* from);
	void	SortNodeLists();
	void	ClearNodeLists();
	void	DrawNodes();
	void	DrawNode(SceneNode*n);

	SceneNode*	root;
	Camera*		camera;
	Shader*		simpleShader;
	Shader*		skyboxShader;
	Shader*		reflectShader;
	Shader*		sceneShader;
	Shader*		pointlightShader;
	Shader*		combineShader;
	Shader*		textShader;

	Frustum		frameFrustum;

	vector<SceneNode*> transparentNodeList;
	vector<SceneNode*> nodeList;

	Font*	basicFont;

	static Renderer*	instance;

	float speed;
};

