#pragma once
#include <vector>
#include <math.h>
#include <time.h>
#include <iostream>

#define MSE 1

#define TANH 1
#define SIGMOID 2

class NeuralNetwork
{
private:

public:

	std::vector<int> node_list;
	std::vector<int> activation_function_list;

	float** original_value_lists;
	float** value_lists;
	float** weight_lists;
	float** bias_lists;

	NeuralNetwork();
	~NeuralNetwork();

	//activation functions
	float tanh_prime(float n);
	float sigmoid(float n);
	float sigmoid_prime(float n);

	void add_layer(int nodes, int activation_function = 0);
	void create();
	void copy_output(float* out);
	void randomise_network();

	float* forward(float* input);

	void train(float* input, float* answer, const int loss_function, const float learning_rate);
};
