#define _CRT_SECURE_NO_WARNINGS
#define USE_CONSOLE

#include <stdlib.h>
#include "NN.h"
#include <stdio.h>
#include <SFML/Graphics.hpp>
#include <ctype.h>

const int nLearn = 60000;
const int nTest = 10000;

inline void clear()
{
	fseek(stdin, 0, SEEK_END);
}

//teaching new network
void teaching(NNetwork *net, double **l_in, double **l_out)
{
	char str[100];
	char choice;
	double l_rate;
	int packet;
	printf("learning rate: ");
	clear();
	while (!scanf("%lf", &l_rate))
		clear();
	printf("number of samples in a packet: ");
	clear();
	while (!scanf("%d", &packet))
		clear();

	learn(net, nLearn * 9 / 10, l_rate, packet, l_in, l_out, nLearn / 10, l_in + nLearn * 9 / 10, l_out + nLearn * 9 / 10, 1, 100);

	//saving the network
	printf("\n");
	printf("Do you want to save trained network?(y/n):");
	choice = 0;
	clear();
	while (choice != 'Y' && choice != 'N')
	{
		scanf("%c", &choice);
		choice = toupper(choice);
	}
	if (choice == 'Y')
	{
		clear();
		printf("Give file name:");
		while (!scanf("%s", str))
			clear();
		if (save(net, str))
			printf("Couldn't save the file\n");
	}
}

//reading network from file
int reading(NNetwork *net)
{
	char str[100];
	clear();
	printf("Give file name:");
	while (!scanf("%s", str))
		clear();
	if (read(net, str))
	{
		printf("Couldn't read the file\n");
		return 1;
	}
	return 0;
}

//examining loaded network
void exam(NNetwork *net, double **t_in, double **t_out)
{
	int n = 0;
	int correct = 1;
	int number;
	printf("how many test samples do you want to see?: ");
	clear();
	while (!scanf("%d", &number))
		clear();

	//initializing allegro, creating window with dimensions 280/280

	while (n < number)
	{
		sf::RenderWindow window(sf::VideoMode(280, 280), "digit");
		sf::Image image;
		image.create(280, 280, sf::Color(0, 0, 0));
		sf::Texture texture;
		texture.create(280, 280);
		//load in test sample and feed forward
		for (int i = 0; i < 28 * 28; i++)
			net->nodes[0][i] = t_in[n][i];
		work(net);

		//check response of neural network, choose digit with highest probability
		int max = 0;
		double maximum = net->nodes[net->nLayers - 1][0];
		for (int i = 1; i < 10; i++)
			if (maximum < net->nodes[net->nLayers - 1][i])
			{
				max = i;
				maximum = net->nodes[net->nLayers - 1][i];
			}

		//check if guess was correct
		if (t_out[n][max])
			correct = 1;
		else
			correct = 0;

		printf("\r%d: %.1f%%                ", max, maximum * 100);

		//draw a picture of given digit
		sf::Color color;
		for (int i = 0; i < 28; i++)
			for (int j = 0; j < 28; j++)
			{
				for (int k = 0; k < 10; k++)
					for (int l = 0; l < 10; l++)
					{
						if (correct)
							color = sf::Color(t_in[n][i * 28 + j] * 255, t_in[n][i * 28 + j] * 255, t_in[n][i * 28 + j] * 255);
						else
							color = sf::Color(255, t_in[n][i * 28 + j] * 255, t_in[n][i * 28 + j] * 255);
						image.setPixel(j * 10 + l, i * 10 + k, color);
					}
			}
		n++;

		//wait for next test sample
		texture.update(image);
		sf::Sprite sprite(texture);
		while(window.isOpen())
		{
			sf::Event event;
			while(window.pollEvent(event))
			{
				if(event.type == sf::Event::Closed)
					window.close();
			}
			window.clear();
			window.draw(sprite);
			window.display();
		}

	}
}

//loading data samples
int load(double **l_in, double **l_out, double **t_in, double **t_out, const char *n1, const char *n2, const char *n3, const char *n4)
{
	const int size = 28 * 28;

	//some auxilary variables
	unsigned char pixels[size];
	unsigned char digit;


	//loading training and test data
	FILE *fTrainImg = fopen(n1, "rb");
	FILE *fTrainLab = fopen(n2, "rb");
	FILE *fTestImg = fopen(n3, "rb");
	FILE *fTestLab = fopen(n4, "rb");
	if (fTrainImg == NULL || fTrainLab == NULL || fTestImg == NULL || fTestLab == NULL)
		return 1;
	fseek(fTrainImg, 16, SEEK_SET);
	fseek(fTrainLab, 8, SEEK_SET);
	fseek(fTestImg, 16, SEEK_SET);
	fseek(fTestLab, 8, SEEK_SET);

	//creating arrays for inputs and outputs from files
	for (int i = 0; i < nLearn; i++)
	{
		fread(pixels, sizeof(unsigned char), 28 * 28, fTrainImg);
		l_in[i] = (double*)malloc(size * sizeof(double));
		for (int j = 0; j < size; j++)
		{
			l_in[i][j] = double(pixels[j]) / 255.0;
		}
		fread(&digit, sizeof(unsigned char), 1, fTrainLab);
		l_out[i] = (double*)malloc(10 * sizeof(double));
		for (int j = 0; j < 10; j++)
		{
			if (j == digit)
				l_out[i][j] = 1;
			else
				l_out[i][j] = 0;
		}
	}
	fclose(fTrainImg);
	fclose(fTrainLab);
	for (int i = 0; i < nTest; i++)
	{
		fread(pixels, sizeof(unsigned char), 28 * 28, fTestImg);
		t_in[i] = (double*)malloc(size * sizeof(double));
		for (int j = 0; j < size; j++)
		{
			t_in[i][j] = double(pixels[j]) / 255;
		}
		fread(&digit, sizeof(unsigned char), 1, fTestLab);
		t_out[i] = (double*)malloc(10 * sizeof(double));
		for (int j = 0; j < 10; j++)
		{
			if (j == digit)
				t_out[i][j] = 1;
			else
				t_out[i][j] = 0;
		}
	}
	fclose(fTestImg);
	fclose(fTestLab);

	printf("training and test data loaded\n");
	return 0;
}

int main()
{
	//initializing network with 4 layers, 784 inputs, two hidden with 16 neurons and 10 output ones
	NNetwork *net = new NNetwork;
	initialize(net, 4, 28 * 28, 16, 16, 10);

	double **l_in = (double **)malloc(nLearn * sizeof(double*));
	double **l_out = (double **)malloc(nLearn * sizeof(double*));
	double **t_in = (double **)malloc(nTest * sizeof(double*));
	double **t_out = (double **)malloc(nTest * sizeof(double*));
	
	if (load(l_in, l_out, t_in, t_out, "train_img", "train_labels", "test_images", "test_labels"))
	{
		printf("Couldn't load training and test images\n");
		return 0;
	}

	char init = 0;
	char choice = 0;

	while (true)
	{
		if (init)
		{
			printf("\nChoose: learning(l), reading(r), examining(e), terminate(t): ");
			while (choice != 'L' && choice != 'R' && choice != 'E' && choice != 'T')
			{
				clear();
				scanf("%c", &choice);
				choice = toupper(choice);
			}
		}
		else
		{
			printf("\nChoose: learning(l), reading(r), terminate(t): ");
			while (choice != 'L' && choice != 'R' && choice != 'T')
			{
				clear();
				scanf("%c", &choice);
				choice = toupper(choice);
			}
		}
		if (choice == 'T')
			break;
		switch (choice)
		{
		case 'L':
			teaching(net, l_in, l_out);
			init = 1;
			break;
		case 'R':
			if (reading(net))
				init = 0;
			else
				init = 1;
			break;
		case 'E':
			exam(net, t_in, t_out);
		}
		choice = 0;
	}

	//deallocate everything and finish
	for (int i = 0; i < nLearn; i++)
	{
		free(l_in[i]);
		free(l_out[i]);
	}
	for (int i = 0; i < nTest; i++)
	{
		free(t_in[i]);
		free(t_out[i]);
	}
	free(l_in);
	free(l_out);
	free(t_in);
	free(t_out);


	destroy(net);
	delete net;
	return 0;
}