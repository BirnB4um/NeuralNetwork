#pragma once
#include <vector>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>

#define e 2.7182818284
#define pi_ 3.141592653

//loss functions
#define MSE 101
#define BI_CROSS_ENTROPY 102

//activation functions
#define TANH 1
#define SIGMOID 2
#define RELU 3

class NeuralNetwork
{
private:
	bool network_created;
	int last_layer_index;
	bool output_error_to_file;
	float* all_gradients;
	std::string output_error_file;
	std::ofstream error_file;

	//int total_weight_list_size;
	//int total_bias_list_size;
	void delete_data();

public:

	std::vector<int> node_list;
	std::vector<int> activation_function_list;

	float** original_value_lists;
	float** value_lists;
	float** weight_lists;
	float** bias_lists;

	NeuralNetwork();
	~NeuralNetwork();

	void set_output_error_to_file(bool should_ouput_to_file, std::string file);

	//activation functions
	float tanh_prime(float n);
	float sigmoid(float n);
	float sigmoid_prime(float n);

	//loss functions
	float loss_prime(const int loss_function, float* answer, float* gradient, const bool add_to_gradient);

	float get_current_error(float* answer, const int loss_function);

	void add_layer(int nodes, int activation_function = 0);
	void create();
	void copy_output(float* out);
	void randomise_network(float min = -1.0f, float max = 1.0f);

	float* forward(float* input);
	float* forward_from_layer(float* input, int layer_index);
	void backward(float learning_rate);
	float* get_output_from_layer(int layer_index);

	float train_once(float* input, float* answer, const int loss_function, const float learning_rate = 0.001);
	/*
	training a batch happens on a single thread
	*/
	float train_batch(float** input, float** answer, const int batch_length, const int loss_function, const float learning_rate = 0.001);
	void train_list(float** input, float** answer, const int list_length, const int loss_function, const float learning_rate = 0.001, const int epoches = 1, const bool print_to_console = true);

	/*
	start_layer must be smaller than end_layer
	returns false if something failed
	*/
	bool cut_network(int start_layer, int end_layer);

	void save_to_file(std::string file);
	bool load_from_file(std::string file);
};
