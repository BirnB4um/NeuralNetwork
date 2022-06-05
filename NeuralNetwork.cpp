#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() {
}
NeuralNetwork::~NeuralNetwork() {
	delete[] value_lists;
	delete[] original_value_lists;
	delete[] weight_lists;
	delete[] bias_lists;
}

void NeuralNetwork::copy_output(float* out) {
	memcpy(out, value_lists[node_list.size() - 1], sizeof(float) * node_list[node_list.size() - 1]);
}

float NeuralNetwork::tanh_prime(float n) {
	n = tanh(n);
	return 1 - n * n;
}
float NeuralNetwork::sigmoid(float n) {
	return n / (1 + abs(n));
}
float NeuralNetwork::sigmoid_prime(float n) {
	n = 1 + abs(n);
	return 1 / (n * n);
}

void NeuralNetwork::add_layer(int nodes, int activation_function) {
	node_list.push_back(nodes);
	activation_function_list.push_back(activation_function);
}

void NeuralNetwork::create() {
	activation_function_list.erase(activation_function_list.end() - 1);

	value_lists = new float* [node_list.size()];
	original_value_lists = new float* [node_list.size()];
	for (int i = 0; i < node_list.size(); i++) {
		value_lists[i] = new float[node_list[i]];
		original_value_lists[i] = new float[node_list[i]];
	}

	weight_lists = new float* [node_list.size() - 1];
	for (int i = 0; i < node_list.size() - 1; i++) {
		weight_lists[i] = new float[node_list[i] * node_list[i + 1]];
	}

	bias_lists = new float* [node_list.size() - 1];
	for (int i = 0; i < node_list.size() - 1; i++) {
		bias_lists[i] = new float[node_list[i + 1]];
	}

	randomise_network();
}

void NeuralNetwork::randomise_network() {
	for (int i = 0; i < node_list.size() - 1; i++) {
		//weights
		for (int n = 0; n < node_list[i] * node_list[i + 1]; n++) {
			weight_lists[i][n] = (float(rand()) / RAND_MAX) * 2 - 1;
		}
		//biases
		for (int n = 0; n < node_list[i + 1]; n++) {
			bias_lists[i][n] = (float(rand()) / RAND_MAX) * 2 - 1;
		}
	}
}

float* NeuralNetwork::forward(float* input) {
	memcpy(value_lists[0], input, node_list[0] * sizeof(float));//copy input layer

	//through layers
	for (int layer_index = 1; layer_index < node_list.size(); layer_index++) {
		//through nodes
		int lower_layer = layer_index - 1;
		for (int node_index = 0; node_index < node_list[layer_index]; node_index++) {
			double sum = 0;
			//weights * inputs
			for (int prev_node_index = 0; prev_node_index < node_list[lower_layer]; prev_node_index++) {
				sum += value_lists[lower_layer][prev_node_index] * weight_lists[lower_layer][node_index * node_list[lower_layer] + prev_node_index];
			}

			//bias
			sum += bias_lists[lower_layer][node_index];

			original_value_lists[layer_index][node_index] = float(sum);

			//activation function
			if (activation_function_list[lower_layer] == TANH) {
				sum = tanh(sum);
			}
			else if (activation_function_list[lower_layer] == SIGMOID) {
				sum = sigmoid(sum);
			}

			value_lists[layer_index][node_index] = float(sum);
		}
	}
	return value_lists[node_list.size() - 1];
}

void NeuralNetwork::train(float* input, float* answer, const int loss_function, const float learning_rate) {
	forward(input);
	int last_layer = node_list.size() - 1;

	float error = 0;
	for (int i = 0; i < node_list[last_layer]; i++) {
		error += (answer[i] - value_lists[last_layer][i]) * (answer[i] - value_lists[last_layer][i]);
	}
	error /= node_list[last_layer];
	//std::cout << error << std::endl;

	//loss_prime
	if (loss_function == MSE) {
		for (int i = 0; i < node_list[last_layer]; i++) {
			value_lists[last_layer][i] = 2 * (value_lists[last_layer][i] - answer[i]) / node_list[last_layer];
		}
	}
	else {
		return;
	}

	//backward
	for (int layer_index = node_list.size() - 1; layer_index > 0; layer_index--) {
		//activationfunction inverse
		if (activation_function_list[layer_index - 1] == TANH) {
			for (int i = 0; i < node_list[layer_index]; i++) {
				value_lists[layer_index][i] *= tanh_prime(original_value_lists[layer_index][i]);
			}
		}
		else if (activation_function_list[layer_index - 1] == SIGMOID) {
			for (int i = 0; i < node_list[layer_index]; i++) {
				value_lists[layer_index][i] *= sigmoid_prime(original_value_lists[layer_index][i]);
			}
		}

		//dense inverse
		for (int input_index = 0; input_index < node_list[layer_index - 1]; input_index++) {
			float input = value_lists[layer_index - 1][input_index];

			//calc next gradient
			value_lists[layer_index - 1][input_index] = 0;
			for (int i = 0; i < node_list[layer_index]; i++) {
				value_lists[layer_index - 1][input_index] += value_lists[layer_index][i] * weight_lists[layer_index - 1][i * node_list[layer_index - 1] + input_index];
			}

			for (int i = 0; i < node_list[layer_index]; i++) {
				//calc weights
				weight_lists[layer_index - 1][i * node_list[layer_index - 1] + input_index] -= learning_rate * value_lists[layer_index][i] * input;

				//calc bias
				bias_lists[layer_index - 1][i] -= learning_rate * value_lists[layer_index][i];
			}
		}
	}
}