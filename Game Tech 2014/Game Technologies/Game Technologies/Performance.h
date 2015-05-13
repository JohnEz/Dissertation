#pragma once

class Performance {
public:
	static Performance* GetInstance();

	void calculateFPS(float msec);
	void calculatePPS(float msec);

	float getFPS() { return fps; };
	float getPPS() { return PPS; };

	int		getCollisions() { return collisions; };
	void	setCollisions(int val) { collisions = val; };
	void	addCollisions() { collisions++; };

	int		getScore() { return score; };
	void	setScore(int val) { score = val; };
	void	addScore(int val) { score += val; };

	float		getAveragePPS() { return averagePPS; };
	float		getMinPPS() { return minPPS; };
	float		getMaxPPS() { return maxPPS; };

protected:
	Performance();
	~Performance();

	static Performance* pInst;

	int		frameCountFPS;
	float	currentTimeFPS;
	float	previousTimeFPS;
	float	fps;

	int		frameCountPPS;
	float	currentTimePPS;
	float	previousTimePPS;
	float	PPS;
	float	maxPPS;
	float	minPPS;
	float	averagePPS;
	float	overallPPS;
	int		PPSCount;

	int		collisions;

	int score;
};