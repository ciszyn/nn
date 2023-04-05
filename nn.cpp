#define _CRT_SECURE_NO_WARNINGS

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include "NN.h"

//function used for shuffling learning data for next epoch
void shuffle(double **array1, double **array2, int n)
{
	for (int i = 0; i < n - 1; i++)
	{
		int j = i + rand() / (RAND_MAX / (n - i) + 1);
		double* t = array1[j];
		array1[j] = array1[i];
		array1[i] = t;
		t = array2[j];
		array2[j] = array2[i];
		array2[i] = t;
	}
}

//function for generating random numbers with normal distribution
double randn(double mu, double sigma)
{
	double U1, U2, W, mult;
	static double X1, X2;
	static int call = 0;

	if (call == 1)
	{
		call = !call;
		return (mu + sigma * (double)X2);
	}

	do
	{
		U1 = -1 + ((double)rand() / RAND_MAX) * 2;
		U2 = -1 + ((double)rand() / RAND_MAX) * 2;
		W = pow(U1, 2) + pow(U2, 2);
	} while (W >= 1 || W == 0);

	mult = sqrt((-2 * log(W)) / W);
	X1 = U1 * mult;
	X2 = U2 * mult;

	call = !call;

	return (mu + sigma * (double)X1);
}

//relu squishification function
double RELU(double x)
{
	if (x > 0)
		return x;
	else
		return 0.1 * x;
}

//relu derivative
double relu_prim(double x)
{
	if (x > 0)
		return 1;
	else
		return 0.1;
}

//function for initializing neural network
void initialize(NNetwork *n, int num, ...)
{
	n->nLayers = num;
	va_list arguments;
	va_start(arguments, num);

	//allocating all arrays
	n->nNodes = (int *)malloc(num * sizeof(int));
	n->nodes = (double **)malloc(num * sizeof(double *));
	for (int i = 0; i < num; i++)
	{
		n->nNodes[i] = va_arg(arguments, int);
		n->nodes[i] = (double *)malloc(n->nNodes[i] * sizeof(double));
	}
	n->biases = (double **)malloc((num - 1) * sizeof(double *));
	n->weights = (double ***)malloc((num - 1) * sizeof(double ***));
	for (int i = 0; i < num - 1; i++)
	{
		n->biases[i] = (double *)malloc(n->nNodes[i] * sizeof(double));
		n->weights[i] = (double **)malloc(n->nNodes[i] * sizeof(double **));
		for (int j = 0; j < n->nNodes[i]; j++)
			n->weights[i][j] = (double *)malloc(n->nNodes[i + 1] * sizeof(double));
	}

	//choosing random values for the network at the beginning
	srand(time(0));
	for (int i = 0; i < num - 1; i++)
		for (int j = 0; j < n->nNodes[i]; j++)
		{
			n->biases[i][j] = randn(0, 1) * sqrt(2.0 / n->nNodes[i]);
			for (int k = 0; k < n->nNodes[i + 1]; k++)
			{
				n->weights[i][j][k] = randn(0, 1) * sqrt(2.0 / n->nNodes[i]);
			}
		}
	va_end(arguments);
}

//function for deallocating memory of neural network
void destroy(NNetwork *n)
{
	if (!n->nLayers)
		return;
	//destroing weights and biases
	for (int i = 0; i < n->nLayers - 1; i++)
		for (int j = 0; j < n->nNodes[i]; j++)
			free(n->weights[i][j]);
	for (int i = 0; i < n->nLayers - 1; i++)
	{
		free(n->weights[i]);
		free(n->biases[i]);
	}
	free(n->weights);
	free(n->biases);

	//destroing nodes
	for (int i = 0; i < n->nLayers; i++)
	{
		free(n->nodes[i]);
	}
	free(n->nodes);

	//destroing nNodes
	free(n->nNodes);
	n->nLayers = 0;
}

//sigmoid function used to map values from -inf to inf between 0 and 1
double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

//function to propagate feedforward
//inputs must be given in the first layer
//outputs are stored in the last layer
void work(NNetwork *n)
{
	for (int i = 0; i < n->nNodes[0]; i++)
		n->nodes[0][i] += n->biases[0][i];
	for (int i = 1; i < n->nLayers; i++)
		for (int j = 0; j < n->nNodes[i]; j++)
		{
			double sum = 0;
			for (int k = 0; k < n->nNodes[i - 1]; k++)
			{
				sum += n->nodes[i - 1][k] * n->weights[i-1][k][j];
			}
			if (i != (n->nLayers - 1))
			{
				sum += n->biases[i][j];
				n->nodes[i][j] = RELU(sum);
			}
			//different activation function for last layer
			else
				n->nodes[i][j] = sigmoid(sum);
		}
}

//function for saving a network
int save(NNetwork *n, const char *name)
{
	FILE *f = fopen(name, "wb");
	if (f == NULL)
		return 1;
	fwrite(&n->nLayers, sizeof(int), 1, f);
	fwrite(n->nNodes, sizeof(int), n->nLayers, f);
	for(int i = 0; i < n->nLayers-1; i++)
		for(int j = 0 ; j < n->nNodes[i]; j++)
			fwrite(n->weights[i][j], sizeof(double), n->nNodes[i+1], f);
	for (int i = 0; i < n->nLayers - 1; i++)
		fwrite(n->biases[i], sizeof(double), n->nNodes[i], f);
	fclose(f);
	return 0;
}

//function for restoring a network from file
int read(NNetwork *n, const char *name)
{

	FILE *f = fopen(name, "rb");
	if (f == NULL)
		return 1;
	destroy(n);
	int num;
	fread(&num, sizeof(int), 1, f);
	n->nLayers = num;

	//allocating all arrays

	n->nNodes = (int *)malloc(num * sizeof(int));
	n->nodes = (double **)malloc(num * sizeof(double *));
	n->biases = (double **)malloc((num - 1) * sizeof(double *));
	fread(n->nNodes, sizeof(int), num, f);
	for (int i = 0; i < num; i++)
	{
		n->nodes[i] = (double *)malloc(n->nNodes[i] * sizeof(double));
	}
	n->weights = (double ***)malloc((num - 1) * sizeof(double ***));
	for (int i = 0; i < num - 1; i++)
	{
		n->biases[i] = (double *)malloc(n->nNodes[i] * sizeof(double));
		n->weights[i] = (double **)malloc(n->nNodes[i] * sizeof(double **));
		for (int j = 0; j < n->nNodes[i]; j++)
			n->weights[i][j] = (double *)malloc(n->nNodes[i + 1] * sizeof(double));
	}


	srand(time(0));
	for (int i = 0; i < num - 1; i++)
		for (int j = 0; j < n->nNodes[i]; j++)
			fread(n->weights[i][j], sizeof(double), n->nNodes[i + 1], f);
	for (int i = 0; i < num - 1; i++)
		fread(n->biases[i], sizeof(double), n->nNodes[i], f);

	fclose(f);
	return 0;
}

//function implementing backpropagation
//inputs and outputs contain num arrays of example inputs and corresponding to them proper outputs 
void backprop(NNetwork *n, int num, double **inputs, double **outputs, double l_rate)
{
	//allocating arrays for errors
	double **errors = (double **)malloc(n->nLayers * sizeof(double *));
	for (int i = 0; i < n->nLayers; i++)
		errors[i] = (double *)malloc(n->nNodes[i] * sizeof(double));
	double **delta = (double **)malloc(n->nLayers * sizeof(double *));

	//initializing errors as 0
	for (int i = 0; i < n->nLayers; i++)
	{
		delta[i] = (double *)malloc(n->nNodes[i] * sizeof(double));
		for (int j = 0; j < n->nNodes[i]; j++)
			delta[i][j] = 0;
	}

	//start of backpropagation
	for (int i = 0; i < num; i++)
	{
		for (int j = 0; j < n->nNodes[0]; j++)
			n->nodes[0][j] = inputs[i][j];

		//feed forward
		work(n);

		//calculating errors in last layer as gradient of error function times derivative of activation function
		//error function C=(x-a)^2, activation function ofr last layer - sigmoid
		//for other layers RELU
		for (int j = 0; j < n->nNodes[n->nLayers - 1]; j++)
			errors[n->nLayers - 1][j] = (n->nodes[n->nLayers - 1][j] - outputs[i][j]) * n->nodes[n->nLayers - 1][j] * (1 - n->nodes[n->nLayers - 1][j]);

		//error for other layers: ((w^l)^T * d^(l+1)) cdot RELU'
		for (int j = n->nLayers - 2; j >= 0; j--)
		{
			for (int k = 0; k < n->nNodes[j]; k++)
			{
				double sum = 0;
				for (int l = 0; l < n->nNodes[j + 1]; l++)
					sum += errors[j + 1][l] * n->weights[j][k][l];
				errors[j][k] = sum * relu_prim(n->nodes[j][k]);
			}
		}

		//actualization of error for whole packet
		for (int j = 0; j < n->nLayers; j++)
			for (int k = 0; k < n->nNodes[j]; k++)
				delta[j][k] += errors[j][k];
	}

	//change of biases and weights with respect to derivative of cost and learning rate
	for (int i = 0; i < n->nLayers - 1; i++)
		for (int j = 0; j < n->nNodes[i]; j++)
			n->biases[i][j] -= l_rate / num * delta[i][j];
	for (int i = 0; i < n->nLayers - 1; i++)
		for (int j = 0; j < n->nNodes[i]; j++)
			for (int k = 0; k < n->nNodes[i + 1]; k++)
				n->weights[i][j][k] -= l_rate / num * n->nodes[i][j] * delta[i + 1][k];

	//deallocating used arrays
	for (int i = 0; i < n->nLayers; i++)
		free(delta[i]);
	free(delta);
	for (int i = 0; i < n->nLayers; i++)
		free(errors[i]);
	free(errors);
}

//function handling learning process using backpropagation to do so
void learn(NNetwork *n, int num1, double l_rate, int packet, double **l_in, double **l_out, int num2, double **t_in, double **t_out, int tolerance, int max)
{
	//counts how many times performance became worse
	int counter = 0;
	
	//calculating performance on unknown data
	double t_err_1 = 0;
	for (int i = 0; i < num2; i++)
	{
		for (int j = 0; j < 28 * 28; j++)
			n->nodes[0][j] = t_in[i][j];
		work(n);
		//check response of neural network, choose digit with highest probability
		int max = 0;
		double maximum = n->nodes[n->nLayers - 1][0];
		for (int i = 1; i < 10; i++)
			if (maximum < n->nodes[n->nLayers - 1][i])
			{
				max = i;
				maximum = n->nodes[n->nLayers - 1][i];
			}

		//check if guess was correct
		if (!t_out[i][max])
			t_err_1 += 1;
	}
	t_err_1 /= num2;

	int i = 0;
	while (i < max)
	{
		//execute backpropagation for all learning data divided into packets
		for (int i = 0; i < num1 / packet; i++)
		{
			backprop(n, packet, l_in + i * packet, l_out + i * packet, l_rate);
			if (i % (num1 / packet / 10) == 0)
				printf(".");
		}

		//before next epoch shuffle learning data
		shuffle(l_in, l_out, num1);
		printf("\r                                                                   ");
		printf("\r");
		printf("epoch = %d, ", i++);

		//check performance once more on validation samples
		double t_err_2 = 0;
		for (int i = 0; i < num2; i++)
		{
			for (int j = 0; j < 28 * 28; j++)
				n->nodes[0][j] = t_in[i][j];
			work(n);

			//check response of neural network, choose digit with highest probability
			int max = 0;
			double maximum = n->nodes[n->nLayers - 1][0];
			for (int i = 1; i < 10; i++)
				if (maximum < n->nodes[n->nLayers - 1][i])
				{
					max = i;
					maximum = n->nodes[n->nLayers - 1][i];
				}

			//check if guess was correct
			if (!t_out[i][max])
				t_err_2 += 1;
		}
		t_err_2 /= num2;
		printf("error = %.1f%%     ", t_err_2 * 100);
		
		//if performance is worse then increment counter
		if (t_err_2 > t_err_1)
			counter++;
		else
			counter = 0;

		//if we are overfitting then abort learning
		if (counter == tolerance)
			break;
		t_err_1 = t_err_2;
		
	}
	printf("\n");
}