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

	int		collisions;

	int score;
};