#include "Performance.h"
#include <stdio.h>


//sets instance pointer to null
Performance* Performance::pInst = 0;

Performance* Performance::GetInstance()
{
	if (pInst == 0)
	{
		pInst = new Performance();
	}

	return pInst;
}

Performance::Performance()
{
	frameCountFPS = 0;
	currentTimeFPS = 0;
	previousTimeFPS = 0;
	fps = 0;

	score = 0;

	frameCountPPS = 0;
	currentTimePPS = 0;
	previousTimePPS = 0;
	PPS = 0;
	maxPPS = 0;
	minPPS = 1000000;
	averagePPS = 0;
	PPSCount = 0;
	overallPPS = 0;


	collisions = 0;
}

void Performance::calculateFPS(float msec)
{
	frameCountFPS++;

	currentTimeFPS += msec;

	if (currentTimeFPS > 1000)
	{
		fps = (frameCountFPS  / (currentTimeFPS / 1000));

		currentTimeFPS = 0;

		frameCountFPS = 0;

	}
}

void Performance::calculatePPS(float msec)
{
	frameCountPPS++;

	currentTimePPS += msec;

	if (currentTimePPS > 1000)
	{
		PPS = (frameCountPPS  / (currentTimePPS / 1000));

		currentTimePPS = 0;

		frameCountPPS = 0;

		++PPSCount;
		overallPPS += PPS;
		averagePPS = overallPPS / PPSCount;

		if (PPS < minPPS && PPS != 0)
		{
			minPPS = PPS;
		}

		if (PPS > maxPPS)
		{
			maxPPS = PPS;
		}
	}
}