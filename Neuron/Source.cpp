#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <ctime>
using namespace std;


#define INPUT_NEURONS 4
#define HIDDEN_NEURONS 4
#define OUTPUT_NEURONS 4

double wih[INPUT_NEURONS + 1][HIDDEN_NEURONS];
double who[HIDDEN_NEURONS + 1][OUTPUT_NEURONS];

double inputs[INPUT_NEURONS];
double hidden[HIDDEN_NEURONS];
double target[OUTPUT_NEURONS];
double actual[OUTPUT_NEURONS];

double erro[OUTPUT_NEURONS];
double errh[HIDDEN_NEURONS];

#define LEARN_RATE 0.2
#define RAND_WEIGHT ( ((float)rand() / (float)RAND_MAX) - 0.5 )
#define getSrand() ((float)rand() / (float)RAND_MAX)
#define getRand(x) (int) ((x) * getSrand())

#define sqr(x) ((x) * (x))

typedef struct {
	double health;
	double knife;
	double gun;
	double enemy;
	double out[OUTPUT_NEURONS];
} ELEMENT;

#define MAX_SAMPLES 18

ELEMENT samples[MAX_SAMPLES] = {
	{ 2.0, 0.0, 0.0, 0.0 ,{ 0.0, 0.0, 1.0, 0.0 } },
	{ 2.0, 0.0, 0.0, 1.0 ,{ 0.0, 0.0, 1.0, 0.0 } },
	{ 2.0, 0.0, 1.0, 1.0 ,{ 1.0, 0.0, 0.0, 0.0 } },
	{ 2.0, 0.0, 1.0, 2.0 ,{ 1.0, 0.0, 0.0, 0.0 } },
	{ 2.0, 1.0, 0.0, 2.0 ,{ 0.0, 0.0, 0.0, 1.0 } },
	{ 2.0, 1.0, 0.0, 1.0 ,{ 1.0, 0.0, 0.0, 0.0 } },

	{ 1.0, 0.0, 0.0, 0.0 ,{ 0.0, 0.0, 1.0, 0.0 } },
	{ 1.0, 0.0, 0.0, 1.0 ,{ 0.0, 0.0, 0.0, 1.0 } },
	{ 1.0, 0.0, 1.0, 1.0 ,{ 1.0, 0.0, 0.0, 0.0 } },
	{ 1.0, 0.0, 1.0, 2.0 ,{ 0.0, 0.0, 0.0, 1.0 } },
	{ 1.0, 1.0, 0.0, 2.0 ,{ 0.0, 0.0, 0.0, 1.0 } },
	{ 1.0, 1.0, 0.0, 1.0 ,{ 0.0, 0.0, 0.0, 1.0 } },

	{ 0.0, 0.0, 0.0, 0.0 ,{ 0.0, 0.0, 1.0, 0.0 } },
	{ 0.0, 0.0, 0.0, 1.0 ,{ 0.0, 0.0, 0.0, 1.0 } },
	{ 0.0, 0.0, 1.0, 1.0 ,{ 0.0, 0.0, 0.0, 1.0 } },
	{ 0.0, 0.0, 1.0, 2.0 ,{ 0.0, 1.0, 0.0, 0.0 } },
	{ 0.0, 1.0, 0.0, 2.0 ,{ 0.0, 1.0, 0.0, 0.0 } },
	{ 0.0, 1.0, 0.0, 1.0 ,{ 0.0, 0.0, 0.0, 1.0 } }
};

char *strings[4] = {"Attack", "Run", "Wander", "Hide"};


void assignRandomWeights(void);
double sigmoid(double val);
double sigmoidDerivative(double val);
void feedForward();
void bakcPropagate(void);

int action(double *vector);



int main()
{
	double err;
	int i, sample = 0, iterations = 0, sum = 0, teachIter = 100000;

	FILE *out = fopen("stats.txt", "w");
	bool teach = false;
	if (teach)
	{
		srand((unsigned)time(NULL));
		assignRandomWeights();

		while (1)
		{
			if (++sample == MAX_SAMPLES) sample = 0;

			inputs[0] = samples[sample].health;
			inputs[1] = samples[sample].knife;
			inputs[2] = samples[sample].gun;
			inputs[3] = samples[sample].enemy;

			target[0] = samples[sample].out[0];
			target[1] = samples[sample].out[1];
			target[2] = samples[sample].out[2];
			target[3] = samples[sample].out[3];

			feedForward();

			err = 0.0;
			for (i = 0; i < OUTPUT_NEURONS; i++)
			{
				err += sqr((samples[sample].out[i] - actual[i]));
			}
			err = 0.5 * err;

			fprintf(out, "%g\n", err);
			printf("mse = %g\n", err);

			if (iterations++ > teachIter) break;

			bakcPropagate();
		}
	}
	else
	{
		fstream fileRead;
		fileRead.open("weight2.txt", /*ios::binary |*/ ios::in);
		for (int inp = 0; inp < INPUT_NEURONS + 1; inp++)
		{
			for (int hid = 0; hid < HIDDEN_NEURONS; hid++)
			{
				fileRead >> wih[inp][hid];
			}
		}
		for (int hid = 0; hid < HIDDEN_NEURONS + 1; hid++)
		{
			for (int outp = 0; outp < OUTPUT_NEURONS; outp++)
			{
				fileRead >> who[hid][outp];
			}
		}

		for (int j = 0; j < HIDDEN_NEURONS; j++)
		{
			fileRead >> hidden[j];
		}
		for (int j = 0; j < OUTPUT_NEURONS; j++)
		{
			fileRead >>	target[j];
		}
		for (int j = 0; j < OUTPUT_NEURONS; j++)
		{
			fileRead >>	actual[j];
		}
		for (int j = 0; j < INPUT_NEURONS; j++)
		{
			fileRead >> inputs[j];
		}
		for (int j = 0; j < OUTPUT_NEURONS; j++)
		{
			fileRead >> erro[j];
		}
		for (int j = 0; j < HIDDEN_NEURONS; j++)
		{
			fileRead >> errh[j];
		}
		fileRead.close();
	}
	fclose(out);
	for (i = 0; i < MAX_SAMPLES; i++)
	{
		inputs[0] = samples[i].health;
		inputs[1] = samples[i].knife;
		inputs[2] = samples[i].gun;
		inputs[3] = samples[i].enemy;

		target[0] = samples[i].out[0];
		target[1] = samples[i].out[1];
		target[2] = samples[i].out[2];
		target[3] = samples[i].out[3];

		feedForward();

		if (action(actual) != action(target))
		{
			printf("%2.1g:%2.1g:%2.1g:%2.1g %s (%s)\n", inputs[0], inputs[1], inputs[2], inputs[3], strings[action(actual)], strings[action(target)]);
		}
		else
		{
			sum++;
		}
	}

	printf("Network is %g%% correct\n", ((float)sum / (float)MAX_SAMPLES) * 100.0);

	inputs[0] = 2.0; inputs[1] = 1.0; inputs[2] = 1.0; inputs[3] = 1.0;
	feedForward();
	printf("2111 Action %s\n", strings[action(actual)]);

	inputs[0] = 1.0; inputs[1] = 1.0; inputs[2] = 1.0; inputs[3] = 2.0;
	feedForward();
	printf("1112 Action %s\n", strings[action(actual)]);

	inputs[0] = 0.0; inputs[1] = 0.0; inputs[2] = 0.0; inputs[3] = 0.0;
	feedForward();
	printf("0000 Action %s\n", strings[action(actual)]);

	inputs[0] = 0.0; inputs[1] = 1.0; inputs[2] = 1.0; inputs[3] = 1.0;
	feedForward();
	printf("0111 Action %s\n", strings[action(actual)]);

	inputs[0] = 2.0; inputs[1] = 0.0; inputs[2] = 1.0; inputs[3] = 3.0;
	feedForward();
	printf("2013 Action %s\n", strings[action(actual)]);

	inputs[0] = 2.0; inputs[1] = 1.0; inputs[2] = 0.0; inputs[3] = 3.0;
	feedForward();
	printf("2103 Action %s\n", strings[action(actual)]);

	inputs[0] = 0.0; inputs[1] = 1.0; inputs[2] = 0.0; inputs[3] = 3.0;
	feedForward();
	printf("0103 Action %s\n", strings[action(actual)]);

	inputs[0] = 2.0; inputs[1] = 1.0; inputs[2] = 0.0; inputs[3] = 1.0;
	feedForward();
	printf("2101 Action %s\n", strings[action(actual)]);

	inputs[0] = 1.0; inputs[1] = 0.0; inputs[2] = 1.0; inputs[3] = 1.0;
	feedForward();
	printf("1011 Action %s\n", strings[action(actual)]);

	

	if (teach)
	{
		fstream fileRead;
		fileRead.open("weight2.txt", /*ios::binary |*/ ios::out);
		for (int inp = 0; inp < INPUT_NEURONS + 1; inp++)
		{
			for (int hid = 0; hid < HIDDEN_NEURONS; hid++)
			{
				fileRead << wih[inp][hid] << endl;
			}
		}
		for (int hid = 0; hid < HIDDEN_NEURONS + 1; hid++)
		{
			for (int outp = 0; outp < OUTPUT_NEURONS; outp++)
			{
				fileRead << who[hid][outp] << endl;
			}
		}

		for (int j = 0; j < HIDDEN_NEURONS; j++)
		{
			fileRead << hidden[j] << endl;
		}
		for (int j = 0; j < OUTPUT_NEURONS; j++)
		{
			fileRead << target[j] << endl;
		}
		for (int j = 0; j < OUTPUT_NEURONS; j++)
		{
			fileRead << actual[j] << endl;
		}
		for (int j = 0; j < INPUT_NEURONS; j++)
		{
			fileRead << inputs[j] << endl;
		}
		for (int j = 0; j < OUTPUT_NEURONS; j++)
		{
			fileRead << erro[j] << endl;
		}
		for (int j = 0; j < HIDDEN_NEURONS; j++)
		{
			fileRead << errh[j] << endl;
		}
		fileRead.close();
	}
	else {
		/*for (int inp = 0; inp < INPUT_NEURONS + 1; inp++)
		{
			for (int hid = 0; hid < HIDDEN_NEURONS; hid++)
			{
				printf("wih %f\n", wih[inp][hid]);
			}
		}
		for (int hid = 0; hid < HIDDEN_NEURONS + 1; hid++)
		{
			for (int outp = 0; outp < OUTPUT_NEURONS; outp++)
			{
				printf("who %f\n", who[hid][outp]);
			}
		}

		for (int j = 0; j < HIDDEN_NEURONS; j++)
		{
			printf("hidden %f\n", hidden[j]);
		}
		for (int j = 0; j < OUTPUT_NEURONS; j++)
		{
			printf("target %f\n", target[j]);
		}
		for (int j = 0; j < OUTPUT_NEURONS; j++)
		{
			printf("actual %f\n", actual[j]);
		}*/
	}
	return 0;
}


void assignRandomWeights(void)
{
	int hid, inp, out;

	for (inp = 0; inp < INPUT_NEURONS + 1; inp++)
	{
		for (hid = 0; hid < HIDDEN_NEURONS; hid++)
		{
			wih[inp][hid] = RAND_WEIGHT;
		}
	}

	for (hid = 0; hid < HIDDEN_NEURONS + 1; hid++)
	{
		for (out = 0; out < OUTPUT_NEURONS; out++)
		{
			wih[hid][out] = RAND_WEIGHT;
		}
	}
}

double sigmoid(double val)
{
	return (1.0 / (1.0 + exp(-val)));
}

double sigmoidDerivative(double val)
{
	return (val * (1.0 - val));
}

void feedForward()
{
	int inp, hid, out;
	double sum;

	for (hid = 0; hid < HIDDEN_NEURONS; hid++)
	{
		sum = 0.0;
		for (inp = 0; inp < INPUT_NEURONS; inp++)
		{
			sum += inputs[inp] * wih[inp][hid];
		}

		sum += wih[HIDDEN_NEURONS][hid];
		hidden[hid] = sigmoid(sum);
	}

	for (out = 0; out < OUTPUT_NEURONS; out++)
	{
		sum = 0.0;
		for (hid = 0; hid < HIDDEN_NEURONS; hid++)
		{
			sum += hidden[hid] * who[hid][out];
		}
		
		sum += who[HIDDEN_NEURONS][out];

		actual[out] = sigmoid(sum);
	}
}

void bakcPropagate(void)
{
	int inp, hid, out;

	for (out = 0; out < OUTPUT_NEURONS; out++)
	{
		erro[out] = (target[out] - actual[out]) * sigmoidDerivative(actual[out]);
	}

	for (hid = 0; hid < OUTPUT_NEURONS; hid++)
	{
		errh[hid] = 0.0;
		for (out = 0; out < OUTPUT_NEURONS; out++)
		{
			errh[hid] += erro[out] * who[hid][out];
		}

		errh[hid] *= sigmoidDerivative(hidden[hid]);
	}

	for (out = 0; out < OUTPUT_NEURONS; out++)
	{
		for (hid = 0; hid < HIDDEN_NEURONS; hid++)
		{
			who[hid][out] += (LEARN_RATE * erro[out] * hidden[hid]);
		}
		who[HIDDEN_NEURONS][out] += LEARN_RATE * erro[out];
	}

	for (hid = 0; hid < HIDDEN_NEURONS; hid++)
	{
		for (inp = 0; inp < INPUT_NEURONS; inp++)
		{
			wih[inp][hid] += (LEARN_RATE * errh[hid] * inputs[inp]);
		}

		wih[INPUT_NEURONS][hid] += LEARN_RATE * errh[hid];
	}
}

int action(double *vector)
{
	int index, sel;
	double max;
	sel = 0;
	max = vector[sel];

	for (index = 1; index < OUTPUT_NEURONS; index++)
	{
		if (vector[index] > max)
		{
			max = vector[index];
			sel = index;
		}
	}
	return sel;
}