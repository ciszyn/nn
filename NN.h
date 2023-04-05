#pragma once
struct NNetwork
{
	int nLayers;
	int *nNodes;
	double **nodes;

	double ***weights;
	double **biases;
};

void initialize(NNetwork *n, int num, ...);
void destroy(NNetwork *n);
void work(NNetwork *n);
int save(NNetwork *n, const char *name);
int read(NNetwork *n, const char *name);
void backprop(NNetwork *n, int num, double **inputs, double **outputs, double l_rate);
void learn(NNetwork *n, int num1, double l_rate, int packet, double **l_in, double **l_out, int num2, double **t_in, double **t_out, int tolerance, int max);