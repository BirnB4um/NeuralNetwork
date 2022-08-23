#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() {
	output_error_to_file = false;

	value_lists = nullptr;
	original_value_lists = nullptr;
	weight_lists = nullptr;
	bias_lists = nullptr;
}
NeuralNetwork::~NeuralNetwork() {
	delete_data();
}

void NeuralNetwork::delete_data() {
	if (original_value_lists != nullptr) {
		for (int i = 0; i < node_list.size(); i++) {
			if (original_value_lists[i] != nullptr) {
				delete[] original_value_lists[i];
			}
		}
		delete[] original_value_lists;
	}

	if (value_lists != nullptr) {
		for (int i = 0; i < node_list.size(); i++) {
			if (value_lists[i] != nullptr) {
				delete[] value_lists[i];
			}
		}
		delete[] value_lists;
	}

	if (weight_lists != nullptr) {
		for (int i = 0; i < node_list.size() - 1; i++) {
			if (weight_lists[i] != nullptr) {
				delete[] weight_lists[i];
			}
		}
		delete[] weight_lists;
	}

	if (bias_lists != nullptr) {
		for (int i = 0; i < node_list.size() - 1; i++) {
			if (bias_lists[i] != nullptr) {
				delete[] bias_lists[i];
			}
		}
		delete[] bias_lists;
	}

	last_layer_index = 0;
	node_list.clear();
	activation_function_list.clear();
}

void NeuralNetwork::copy_output(float* out) {
	memcpy(out, value_lists[last_layer_index], sizeof(float) * node_list[last_layer_index]);
}

float NeuralNetwork::tanh_prime(float n) {
	n = tanh(n);
	return 1 - n * n;
}
float NeuralNetwork::sigmoid(float n) {
	n = 1 / std::pow(e, n);
	return 1 / (1 + n);
}
float NeuralNetwork::sigmoid_prime(float n) {
	n = sigmoid(n);
	return n * (1 - n);
}

void NeuralNetwork::set_output_error_to_file(bool should_ouput_to_file, std::string file) {
	output_error_to_file = should_ouput_to_file;
	output_error_file = file;
}

void NeuralNetwork::add_layer(int nodes, int activation_function) {
	node_list.push_back(nodes);
	activation_function_list.push_back(activation_function);
}

void NeuralNetwork::create() {
	//total_weight_list_size = 0;

	activation_function_list.erase(activation_function_list.end() - 1);
	last_layer_index = node_list.size() - 1;

	value_lists = new float* [node_list.size()];
	original_value_lists = new float* [node_list.size()];
	for (int i = 0; i < node_list.size(); i++) {
		value_lists[i] = new float[node_list[i]];
		original_value_lists[i] = new float[node_list[i]];
	}

	weight_lists = new float* [last_layer_index];
	//total_weight_list_size += last_layer_index * 4;
	for (int i = 0; i < last_layer_index; i++) {
		weight_lists[i] = new float[node_list[i] * node_list[i + 1]];
		//total_weight_list_size += node_list[i] * node_list[i + 1] * 4;
	}

	bias_lists = new float* [last_layer_index];
	for (int i = 0; i < last_layer_index; i++) {
		bias_lists[i] = new float[node_list[i + 1]];
	}

	randomise_network();
}

void NeuralNetwork::randomise_network() {
	for (int i = 0; i < last_layer_index; i++) {
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
			else if (activation_function_list[lower_layer] == RELU) {
				sum = sum < 0 ? 0 : sum;
			}

			value_lists[layer_index][node_index] = float(sum);
		}
	}
	return value_lists[last_layer_index];
}

void NeuralNetwork::backward(const float learning_rate) {
	for (int layer_index = last_layer_index; layer_index > 0; layer_index--) {
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
		else if (activation_function_list[layer_index - 1] == RELU) {
			for (int i = 0; i < node_list[layer_index]; i++) {
				value_lists[layer_index][i] *= original_value_lists[layer_index][i] > 0 ? 1 : 0;
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

float NeuralNetwork::get_current_error(float* answer, const int loss_function) {
	int last_layer_size = node_list[last_layer_index];
	float error = 0;
	//loss_prime
	if (loss_function == MSE) {
		for (int i = 0; i < last_layer_size; i++) {
			error += (answer[i] - value_lists[last_layer_index][i]) * (answer[i] - value_lists[last_layer_index][i]);
		}
	}
	else if (loss_function == BI_CROSS_ENTROPY) {
		for (int i = 0; i < last_layer_size; i++) {
			error += (-answer[i] * log(value_lists[last_layer_index][i]) - (1.0f - answer[i]) * log(1.0f - value_lists[last_layer_index][i]));
		}
	}
	else {
		return -1;
	}
	error /= last_layer_size;
	return error;
}

float NeuralNetwork::loss_prime(const int loss_function, float* answer, float* gradient, const bool add_to_gradient) {
	int last_layer_size = node_list[last_layer_index];
	float error = 0;
	//loss_prime
	if (loss_function == MSE) {
		for (int i = 0; i < last_layer_size; i++) {
			error += (answer[i] - value_lists[last_layer_index][i]) * (answer[i] - value_lists[last_layer_index][i]);
			gradient[i] = (2 * (value_lists[last_layer_index][i] - answer[i]) / last_layer_size) + (add_to_gradient ? gradient[i] : 0);
		}
		error /= last_layer_size;
	}
	else if (loss_function == BI_CROSS_ENTROPY) {
		for (int i = 0; i < last_layer_size; i++) {
			//float _1 = -answer[i];
			//float _2 = (1.0f - answer[i]);
			//float log_1 = log(std::abs(value_lists[last_layer_index][i]) + 0.0000001f);
			//float log_2 = log(std::abs(1.0f - value_lists[last_layer_index][i]) + 0.0000001f);
			//float t = (_1 * log_1 - _2 * log_2);

			if (std::isnan(gradient[i]))
				std::cout << gradient[i] << std::endl;
			float grad = value_lists[last_layer_index][i];
			float _1 = value_lists[last_layer_index][i];
			float a = answer[i];
			error += (a * log(1e-15 + _1));

			float _a = (1.0f - answer[i]);
			float _v = (1.0f - value_lists[last_layer_index][i]);
			float a_v = answer[i] / value_lists[last_layer_index][i];
			gradient[i] = (((_a / _v) - a_v) / last_layer_size) + (add_to_gradient ? gradient[i] : 0);

			float grad_ii = value_lists[last_layer_index][i];
			//if (std::isnan(gradient[i]))
			//	std::cout << gradient[i] << std::endl;
		}
		error /= -last_layer_size;
	}
	else {
		return -1;
	}

	return error;
}

float NeuralNetwork::train_once(float* input, float* answer, const int loss_function, const float learning_rate) {
	forward(input);

	//loss_prime
	float error = loss_prime(loss_function, answer, value_lists[last_layer_index], false);
	if (error == -1)
		return -1;

	backward(learning_rate);

	//std::cout << value_lists[last_layer_index][0] << std::endl;

	return error;
}

float NeuralNetwork::train_batch(float** input, float** answer, const int batch_length, const int loss_function, const float learning_rate) {
	int last_layer_size = node_list[last_layer_index];
	float* all_gradients = new float[last_layer_size];

	double error = 0;
	double temp_error = 0;
	for (int batch_index = 0; batch_index < batch_length; batch_index++) {
		forward(input[batch_index]);

		//loss_prime
		temp_error = loss_prime(loss_function, answer[batch_index], all_gradients, true);

		error += temp_error;
	}
	error /= batch_length;

	for (int i = 0; i < last_layer_size; i++) {
		all_gradients[i] /= batch_length;
	}

	//set gradient
	memcpy(value_lists[last_layer_index], all_gradients, sizeof(float) * last_layer_size);
	delete[] all_gradients;

	backward(learning_rate);

	return error;
}

void NeuralNetwork::train_list(float** input, float** answer, const int list_length, const int loss_function, const float learning_rate, int epoches, const bool print_to_console) {
	if (output_error_to_file)
		error_file.open(output_error_file);

	epoches = epoches < 1 ? 1 : epoches;
	for (int epoche = 0; epoche < epoches; epoche++) {
		double error = 0;
		for (int i = 0; i < list_length; i++) {
			error += train_once(input[i], answer[i], loss_function, learning_rate);
		}
		error /= list_length;

		if (output_error_to_file) {
			error_file << error << "\n";
		}

		if (print_to_console)
			std::cout << "epoche " << epoche << "/" << epoches << "   " << "error: " << error << "\n";
	}
	if (output_error_to_file)
		error_file.close();
}

void NeuralNetwork::save_to_file(std::string file) {
	int total_bias_count = 0;
	int total_weight_count = 0;
	for (int layer_index = 1; layer_index < node_list.size(); layer_index++) {
		total_bias_count += node_list[layer_index];
		total_weight_count += node_list[layer_index] * node_list[layer_index - 1];
	}

	int data_size = 4 + //layer count
		4 * node_list.size() + //node list
		4 * (node_list.size() - 1) + //activation function
		4 * total_bias_count + //bias
		4 * total_weight_count; //weight

	uint8_t* output_data = new uint8_t[data_size];

	size_t length = 0;
	size_t where_ = 0;
	int size = node_list.size();

	//layer count
	length = 4;
	memcpy(&(output_data[where_]), &size, length);
	where_ += length;

	//node list
	length = 4 * size;
	memcpy(&(output_data[where_]), &(node_list[0]), length);
	where_ += length;

	//activation function
	length = (size - 1) * 4;
	memcpy(&(output_data[where_]), &(activation_function_list[0]), length);
	where_ += length;

	//bias
	for (int i = 0; i < size - 1; i++) {
		length = 4 * node_list[i + 1];
		memcpy(&(output_data[where_]), &(bias_lists[i][0]), length);
		where_ += length;
	}

	//weight
	for (int i = 0; i < size - 1; i++) {
		length = 4 * node_list[i] * node_list[i + 1];
		memcpy(&(output_data[where_]), &(weight_lists[i][0]), length);
		where_ += length;
	}

	auto myfile = std::fstream(file, std::ios::out | std::ios::binary);
	myfile.write((char*)output_data, data_size);
	myfile.close();
}
void NeuralNetwork::load_from_file(std::string file) {
	std::ifstream input(file, std::ios::binary);
	std::vector<char> file_data = std::vector<char>((std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()));
	input.close();

	delete_data();

	output_error_to_file = false;

	size_t offset = 0;
	size_t length = 0;

	//set number of layers
	int number_of_layers = 0;
	length = 4;
	memcpy(&number_of_layers, &file_data[0], length);
	offset += length;

	last_layer_index = number_of_layers - 1;

	//create node_list
	length = 4;
	for (int i = 0; i < number_of_layers; i++) {
		node_list.push_back(*(int*)&file_data[offset]);
		offset += length;
	}

	//set activation functions
	length = 4;
	for (int i = 0; i < number_of_layers - 1; i++) {
		activation_function_list.push_back(*(int*)&file_data[offset]);
		offset += length;
	}

	//set biases
	bias_lists = new float* [number_of_layers - 1];
	for (int i = 0; i < number_of_layers - 1; i++) {
		bias_lists[i] = new float[node_list[i + 1]];
		length = 4 * node_list[i + 1];
		memcpy(&(bias_lists[i][0]), &(file_data[offset]), length);
		offset += length;
	}

	//set weights
	weight_lists = new float* [number_of_layers - 1];
	for (int i = 0; i < number_of_layers - 1; i++) {
		weight_lists[i] = new float[node_list[i + 1] * node_list[i]];
		length = 4 * node_list[i + 1] * node_list[i];
		memcpy(&(weight_lists[i][0]), &(file_data[offset]), length);
		offset += length;
	}

	value_lists = new float* [node_list.size()];
	original_value_lists = new float* [node_list.size()];
	for (int i = 0; i < node_list.size(); i++) {
		value_lists[i] = new float[node_list[i]];
		original_value_lists[i] = new float[node_list[i]];
	}
}