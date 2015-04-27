#pragma once

#include "./nclgl/Light.h"

#define LIGHTNUM 8 // We ’ll generate LIGHTNUM squared lights ...

class LightStorage
{
public:
	LightStorage(void) { pointLights = new Light[LIGHTNUM * LIGHTNUM]; currentLights = 0; };
	~LightStorage(void) { delete[] pointLights; };

	void addLight(Light l) {
		pointLights[currentLights] = l;
		currentLights++;
	};

	Light* getLights() { return pointLights; }

	static LightStorage* GetInstance() {
		if (lstore == NULL)
		{
			lstore = new LightStorage();
		}

		return lstore;
	};

protected:
	Light* pointLights; // Array of lighting data

	int currentLights;

	//pointer to worldmap instance
	static LightStorage* lstore;
};