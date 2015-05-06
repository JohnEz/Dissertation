#include "Renderer.h"

Renderer* Renderer::instance = NULL;

Renderer::Renderer(Window &parent) : OGLRenderer(parent)	{	
	camera			= NULL;

	root			= new SceneNode();
	speed = 1.0f;
	instance		= this;

	LightStorage::GetInstance();
	quadW = Mesh::GenerateQuad();
	quad = Mesh::GenerateQuad();

	//LightStorage::GetInstance()->addLight(Light(Vector3(100, 100, 100), Vector4(0.9f, 0.9f, 1.0f, 1), (RAW_WIDTH * HEIGHTMAP_X)));
	//LightStorage::GetInstance()->addLight(Light(Vector3(500, 0, 500), Vector4(0.9f, 0.9f, 1.0f, 1), (RAW_WIDTH * HEIGHTMAP_X)));

	basicFont = new Font(SOIL_load_OGL_texture(TEXTUREDIR"tahoma.tga",SOIL_LOAD_AUTO,SOIL_CREATE_NEW_ID,SOIL_FLAG_COMPRESS_TO_DXT),16,16);

	sphere = new OBJMesh();
	if (!sphere->LoadOBJMesh(MESHDIR"ico.obj")) {
		return;
	}

	cubeMap = SOIL_load_OGL_cubemap(TEXTUREDIR"rusted_west.jpg", TEXTUREDIR"rusted_east.jpg", TEXTUREDIR"rusted_up.jpg", TEXTUREDIR"rusted_down.jpg", TEXTUREDIR"rusted_south.jpg", TEXTUREDIR"rusted_north.jpg", SOIL_LOAD_RGB, SOIL_CREATE_NEW_ID, 0);

	if (!cubeMap) {
		return;
	}

	simpleShader	= new Shader(SHADERDIR"TechVertex.glsl", SHADERDIR"TechFragment.glsl");
	if(!simpleShader->LinkProgram() ){
		return;
	}

	skyboxShader = new Shader(SHADERDIR"SkyboxVertex.glsl", SHADERDIR"SkyboxFragment.glsl");
	if (!skyboxShader->LinkProgram()) {
		return;
	}

	sceneShader = new Shader(SHADERDIR"BumpVertex.glsl", SHADERDIR"bufferFragment.glsl");
	if (!sceneShader->LinkProgram()) {
		return;
	}

	reflectShader = new Shader(SHADERDIR"BumpVertex.glsl", SHADERDIR"bufferReflectFragment.glsl");
	if (!reflectShader->LinkProgram()) {
		return;
	}

	combineShader = new Shader(SHADERDIR"combineVertex.glsl", SHADERDIR"combineFragment.glsl");
	if (!combineShader->LinkProgram()) {
		return;
	}

	pointlightShader = new Shader(SHADERDIR"pointlightVertex.glsl", SHADERDIR"pointlightFragment.glsl");
	if (!pointlightShader->LinkProgram()) {
		return;
	}

	skyboxShader = new Shader(SHADERDIR"SkyboxVertex.glsl", SHADERDIR"SkyboxFragment.glsl");
	if (!skyboxShader->LinkProgram()) {
		return;
	}

	textShader = new Shader(SHADERDIR"texturedVertex.glsl", SHADERDIR"texturedFragment.glsl");
	if (!textShader->LinkProgram()) {
		return;
	}

	glGenFramebuffers(1, &bufferFBO);
	glGenFramebuffers(1, &pointLightFBO);

	GLenum buffers[2];
	buffers[0] = GL_COLOR_ATTACHMENT0;
	buffers[1] = GL_COLOR_ATTACHMENT1;

	// Generate our scene depth texture ...
	GenerateScreenTexture(bufferDepthTex, true);
	GenerateScreenTexture(bufferColourTex);
	GenerateScreenTexture(bufferNormalTex);
	GenerateScreenTexture(lightEmissiveTex);
	GenerateScreenTexture(lightSpecularTex);

	// And now attach them to our FBOs
	glBindFramebuffer(GL_FRAMEBUFFER, bufferFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bufferColourTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, bufferNormalTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, bufferDepthTex, 0);
	glDrawBuffers(2, buffers);

	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		return;
	}

	glBindFramebuffer(GL_FRAMEBUFFER, pointLightFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, lightEmissiveTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, lightSpecularTex, 0);
	glDrawBuffers(2, buffers);

	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		return;
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

	init = true;
}

Renderer::~Renderer(void)	{
	delete root;
	delete simpleShader;

	currentShader = NULL;
}

void Renderer::UpdateScene(float msec)	{
	if(camera) {
		camera->UpdateCamera(msec); 
	}

	//Slow and speed growth
	if (Window::GetKeyboard()->KeyDown(KEYBOARD_O))
	{
		speed += 0.01f;
	}
	else if (Window::GetKeyboard()->KeyDown(KEYBOARD_L))
	{
		speed -= 0.01f;
	}

	root->Update(msec * speed);
}

void Renderer::RenderScene()	{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	DrawText("FPS              :" + std::to_string(Performance::GetInstance()->getFPS()), Vector3(0,0,0), 16.0f);
	DrawText("Agent Updates PS :" + std::to_string(Performance::GetInstance()->getPPS()), Vector3(0,16.0f,0), 16.0f);
	DrawText("Agents  :" + std::to_string(Performance::GetInstance()->getScore()), Vector3(0,32.0f,0), 16.0f);
	DrawText("Players :" + std::to_string(Performance::GetInstance()->getCollisions()), Vector3(0,48.0f,0), 16.0f);

	FillBuffers();
	//DrawPointLights();
	CombineBuffers();
	
	SwapBuffers();
}

void	Renderer::DrawNode(SceneNode*n)	{
	if(n->GetMesh()) {
		glUniformMatrix4fv(glGetUniformLocation(currentShader->GetProgram(), "modelMatrix"),	1,false, (float*)&(n->GetWorldTransform()*Matrix4::Scale(n->GetModelScale())));
		glUniform4fv(glGetUniformLocation(currentShader->GetProgram(), "nodeColour"),1,(float*)&n->GetColour());

		n->Draw(*this);
	}
}

void	Renderer::BuildNodeLists(SceneNode* from)	{
	Vector3 direction = from->GetWorldTransform().GetPositionVector() - camera->GetPosition();
	from->SetCameraDistance(Vector3::Dot(direction,direction));

	if(frameFrustum.InsideFrustum(*from)) {
		if(from->GetColour().w < 1.0f) {
			transparentNodeList.push_back(from);
		}
		else{
			nodeList.push_back(from);
		}
	}

	for(vector<SceneNode*>::const_iterator i = from->GetChildIteratorStart(); i != from->GetChildIteratorEnd(); ++i) {
		BuildNodeLists((*i));
	}
}

void	Renderer::DrawNodes()	 {
	for(vector<SceneNode*>::const_iterator i = nodeList.begin(); i != nodeList.end(); ++i ) {
		DrawNode((*i));
	}

	for(vector<SceneNode*>::const_reverse_iterator i = transparentNodeList.rbegin(); i != transparentNodeList.rend(); ++i ) {
		DrawNode((*i));
	}
}

void	Renderer::SortNodeLists()	{
	std::sort(transparentNodeList.begin(),	transparentNodeList.end(),	SceneNode::CompareByCameraDistance);
	std::sort(nodeList.begin(),				nodeList.end(),				SceneNode::CompareByCameraDistance);
}

void	Renderer::ClearNodeLists()	{
	transparentNodeList.clear();
	nodeList.clear();
}

void	Renderer::SetCamera(Camera*c) {
	camera = c;
}

void	Renderer::AddNode(SceneNode* n) {
	root->AddChild(n);
}

void	Renderer::RemoveNode(SceneNode* n) {
	root->RemoveChild(n);
}

void Renderer::DrawSkybox()
{
	glDepthMask(GL_FALSE);
	glDisable(GL_CULL_FACE);

	SetCurrentShader(skyboxShader);

	modelMatrix.ToIdentity();
	viewMatrix = camera->BuildViewMatrix();
	projMatrix = Matrix4::Perspective(1.0f,10000.0f,(float)width / (float)height, 45.0f);

	UpdateShaderMatrices();
	quadW->Draw();

	glUseProgram(0);
	glEnable(GL_CULL_FACE);
	glDepthMask(GL_TRUE);

}

void Renderer::GenerateScreenTexture(GLuint &into, bool depth) {
	glGenTextures(1, &into);
	glBindTexture(GL_TEXTURE_2D, into);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, depth ? GL_DEPTH_COMPONENT24 : GL_RGBA8, width, height, 0, depth ? GL_DEPTH_COMPONENT : GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::FillBuffers() {
	glBindFramebuffer(GL_FRAMEBUFFER, bufferFBO);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	glDisable(GL_BLEND);
	
	DrawSkybox();

	if(camera) {
		//SetCurrentShader(sceneShader);
		SetCurrentShader(simpleShader);
		glUniform1i(glGetUniformLocation(currentShader->GetProgram(), "diffuseTex"), 0);

		textureMatrix.ToIdentity();
		modelMatrix.ToIdentity();
		viewMatrix		= camera->BuildViewMatrix();
		projMatrix		= Matrix4::Perspective(1.0f,10000.0f,(float)width / (float) height, 45.0f);
		frameFrustum.FromMatrix(projMatrix * viewMatrix);
		UpdateShaderMatrices();

		//Return to default 'usable' state every frame!
		glEnable(GL_DEPTH_TEST);
		//glEnable(GL_CULL_FACE);
		glDisable(GL_CULL_FACE);
		glDisable(GL_STENCIL_TEST);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		BuildNodeLists(root);
		SortNodeLists();
		DrawNodes();
		ClearNodeLists();

	}

	//DrawHeightmap();
	//DrawWater();
	//DrawPlant();
	glEnable(GL_BLEND);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::DrawPointLights() {
	SetCurrentShader(pointlightShader);

	glBindFramebuffer(GL_FRAMEBUFFER, pointLightFBO);

	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);

	glBlendFunc(GL_ONE, GL_ONE);

	glUniform1i(glGetUniformLocation(currentShader->GetProgram(), "depthTex"), 3);
	glUniform1i(glGetUniformLocation(currentShader->GetProgram(), "normTex"), 4);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, bufferDepthTex);

	glActiveTexture(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_2D, bufferNormalTex);

	glUniform3fv(glGetUniformLocation(currentShader->GetProgram(), "cameraPos"), 1, (float*)&camera->GetPosition());

	glUniform2f(glGetUniformLocation(currentShader->GetProgram(), "pixelSize"), 1.0f / width, 1.0f / height);

	Vector3 translate = Vector3((RAW_HEIGHT * HEIGHTMAP_X / 2.0f ), 500, (RAW_HEIGHT * HEIGHTMAP_Z / 2.0f));

	Matrix4 pushMatrix = Matrix4::Translation(translate);
	Matrix4 popMatrix = Matrix4::Translation(-translate);

	for (int x = 0; x < LIGHTNUM; ++x) {
		for (int z = 0; z < LIGHTNUM; ++z) {
			Light &l = LightStorage::GetInstance()->getLights()[( x * LIGHTNUM )+ z ];
			float radius = l.GetRadius();

			modelMatrix = pushMatrix * popMatrix * Matrix4::Translation(l.GetPosition()) * Matrix4::Scale(Vector3(radius, radius, radius));

			l.SetPosition(modelMatrix.GetPositionVector());

			SetShaderLight(l);

			UpdateShaderMatrices();

			float dist = (l.GetPosition() - camera->GetPosition()).Length();
			if(dist < radius) 
			{// camera is inside the light volume !
				glCullFace(GL_FRONT);
			}
			else 
			{
				glCullFace(GL_BACK);
			}

			sphere->Draw();
		}
	}
	glCullFace(GL_BACK);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glClearColor(0.2f, 0.2f, 0.2f, 1);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glUseProgram(0);
}

void Renderer::CombineBuffers() {
	SetCurrentShader(combineShader);

	projMatrix = Matrix4::Orthographic(-1, 1, 1, -1, -1, 1);
	UpdateShaderMatrices();

	glUniform1i(glGetUniformLocation(currentShader->GetProgram(), "diffuseTex"), 2);
	glUniform1i(glGetUniformLocation(currentShader->GetProgram(), "emissiveTex"), 3);
	glUniform1i(glGetUniformLocation(currentShader->GetProgram(), "specularTex"), 4);
	glUniform1i(glGetUniformLocation(currentShader->GetProgram(), "normalTex"), 5);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, bufferColourTex);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, lightEmissiveTex);

	glActiveTexture(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_2D, lightSpecularTex);

	glActiveTexture(GL_TEXTURE5);
	glBindTexture(GL_TEXTURE_2D, bufferNormalTex);

	quad->Draw();

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glUseProgram(0);
}

void Renderer::DrawText(const std::string &text, const Vector3 &position, const float size, const bool perspective)	{
	SetCurrentShader(textShader);
	glUniform1i(glGetUniformLocation(textShader->GetProgram(), "diffuseTex"), 0);

	glBlendFunc(GL_SRC_ALPHA,GL_ONE);
	
	//Create a new temporary TextMesh, using our line of text and our font
	TextMesh* mesh = new TextMesh(text,*basicFont);

	//This just does simple matrix setup to render in either perspective or
	//orthographic mode, there's nothing here that's particularly tricky.
	if(perspective) {
		modelMatrix = Matrix4::Translation(position) * Matrix4::Scale(Vector3(size,size,1));
		viewMatrix = camera->BuildViewMatrix();
		projMatrix = Matrix4::Perspective(1.0f,10000.0f,(float)width / (float)height, 45.0f);
	}
	else{	
		//In ortho mode, we subtract the y from the height, so that a height of 0
		//is at the top left of the screen, which is more intuitive
		//(for me anyway...)
		modelMatrix = Matrix4::Translation(Vector3(position.x,height-position.y, position.z)) * Matrix4::Scale(Vector3(size,size,1));
		viewMatrix.ToIdentity();
		projMatrix = Matrix4::Orthographic(-1.0f,1.0f,(float)width, 0.0f,(float)height, 0.0f);
	}
	//Either way, we update the matrices, and draw the mesh
	UpdateShaderMatrices();
	mesh->Draw();

	delete mesh; //Once it's drawn, we don't need it anymore!

	viewMatrix = camera->BuildViewMatrix();

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glUseProgram(0);
}