#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() {
	output_error_to_file = false;
	network_created = false;
	all_gradients = nullptr;
	last_layer_index = -1;

	value_lists = nullptr;
	original_value_lists = nullptr;
	weight_lists = nullptr;
	bias_lists = nullptr;
}

NeuralNetwork::~NeuralNetwork() {
	delete_data();
}

void NeuralNetwork::delete_data() {
	if (!network_created)
		return;

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

	if (all_gradients != nullptr) {
		delete[] all_gradients;
	}

	last_layer_index = 0;
	node_list.clear();
	activation_function_list.clear();
}

void NeuralNetwork::copy_output(float* out) {
	if (!network_created)
		return;
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
	if (network_created)
		return;

	node_list.push_back(nodes);
	activation_function_list.push_back(activation_function);
}

void NeuralNetwork::create() {
	//total_weight_list_size = 0;
	if (network_created)
		return;

	activation_function_list.erase(activation_function_list.begin());
	last_layer_index = node_list.size() - 1;

	value_lists = new float* [node_list.size()];
	original_value_lists = new float* [node_list.size()];
	for (int i = 0; i < node_list.size(); ++i) {
		value_lists[i] = new float[node_list[i]];
		original_value_lists[i] = new float[node_list[i]];
	}

	weight_lists = new float* [last_layer_index];
	//total_weight_list_size += last_layer_index * 4;
	for (int i = 0; i < last_layer_index; ++i) {
		weight_lists[i] = new float[node_list[i] * node_list[i + 1]];
		//total_weight_list_size += node_list[i] * node_list[i + 1] * 4;
	}

	bias_lists = new float* [last_layer_index];
	for (int i = 0; i < last_layer_index; ++i) {
		bias_lists[i] = new float[node_list[i + 1]];
	}

	all_gradients = new float[node_list[last_layer_index]];

	randomise_network(-1.0f, 1.0f);
	network_created = true;
}

void NeuralNetwork::randomise_network(float min, float max) {
	if (!network_created)
		return;

	for (int i = 0; i < last_layer_index; i++) {
		//weights
		for (int n = 0; n < node_list[i] * node_list[i + 1]; ++n) {
			weight_lists[i][n] = min + (float(rand()) / RAND_MAX) * (max-min);
		}
		//biases
		for (int n = 0; n < node_list[i + 1]; ++n) {
			bias_lists[i][n] = min + (float(rand()) / RAND_MAX) * (max - min);
		}
	}
}

float* NeuralNetwork::forward(float* input) {
	if (!network_created)
		return nullptr;

	memcpy(value_lists[0], input, node_list[0] * sizeof(float));//copy input layer

	//through layers
	for (int layer_index = 1; layer_index < node_list.size(); ++layer_index) {
		//through nodes
		int lower_layer = layer_index - 1;
		for (int node_index = 0; node_index < node_list[layer_index]; ++node_index) {
			double sum = 0;
			//weights * inputs
			for (int prev_node_index = 0; prev_node_index < node_list[lower_layer]; ++prev_node_index) {
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

float* NeuralNetwork::forward_from_layer(float* input, int layer_index) {
	if (!network_created)
		return nullptr;

	if (layer_index < 0 || layer_index >= node_list.size()-1) {
		return nullptr;
	}

	memcpy(value_lists[layer_index], input, node_list[layer_index] * sizeof(float));//copy input layer

	//through layers
	for (int layer_index_ = layer_index+1; layer_index_ < node_list.size(); ++layer_index_) {
		//through nodes
		int lower_layer = layer_index_ - 1;
		for (int node_index = 0; node_index < node_list[layer_index_]; ++node_index) {
			double sum = 0;
			//weights * inputs
			for (int prev_node_index = 0; prev_node_index < node_list[lower_layer]; ++prev_node_index) {
				sum += value_lists[lower_layer][prev_node_index] * weight_lists[lower_layer][node_index * node_list[lower_layer] + prev_node_index];
			}

			//bias
			sum += bias_lists[lower_layer][node_index];

			original_value_lists[layer_index_][node_index] = float(sum);

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

			value_lists[layer_index_][node_index] = float(sum);
		}
	}
	return value_lists[last_layer_index];
}

float* NeuralNetwork::get_output_from_layer(int layer_index) {
	if (!network_created)
		return nullptr;

	return value_lists[layer_index];
}

void NeuralNetwork::backward(const float learning_rate) {

	if (!network_created)
		return;

	for (int layer_index = last_layer_index; layer_index > 0; --layer_index) {
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
		for (int input_index = 0; input_index < node_list[layer_index - 1]; ++input_index) {
			float input = value_lists[layer_index - 1][input_index];

			//calc next gradient
			value_lists[layer_index - 1][input_index] = 0;
			for (int i = 0; i < node_list[layer_index]; ++i) {
				value_lists[layer_index - 1][input_index] += value_lists[layer_index][i] * weight_lists[layer_index - 1][i * node_list[layer_index - 1] + input_index];
			}

			for (int i = 0; i < node_list[layer_index]; ++i) {
				//calc weights
				weight_lists[layer_index - 1][i * node_list[layer_index - 1] + input_index] -= learning_rate * value_lists[layer_index][i] * input;

				//calc bias
				bias_lists[layer_index - 1][i] -= learning_rate * value_lists[layer_index][i];
			}
		}
	}
}

float NeuralNetwork::get_current_error(float* answer, const int loss_function) {
	if (!network_created)
		return -1;

	int last_layer_size = node_list[last_layer_index];
	float error = 0;
	//loss_prime
	if (loss_function == MSE) {
		for (int i = 0; i < last_layer_size; ++i) {
			error += (answer[i] - value_lists[last_layer_index][i]) * (answer[i] - value_lists[last_layer_index][i]);
		}
	}
	else if (loss_function == BI_CROSS_ENTROPY) {
		for (int i = 0; i < last_layer_size; ++i) {
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
	if (!network_created)
		return -1;

	int last_layer_size = node_list[last_layer_index];
	float error = 0;
	//loss_prime
	if (loss_function == MSE) {
		for (int i = 0; i < last_layer_size; ++i) {
			error += (answer[i] - value_lists[last_layer_index][i]) * (answer[i] - value_lists[last_layer_index][i]);
			gradient[i] = (2 * (value_lists[last_layer_index][i] - answer[i]) / last_layer_size) + (add_to_gradient ? gradient[i] : 0);
		}
		error /= last_layer_size;
	}
	else if (loss_function == BI_CROSS_ENTROPY) {
		for (int i = 0; i < last_layer_size; ++i) {
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
	if (!network_created)
		return -1;

	forward(input);

	//loss_prime
	float error = loss_prime(loss_function, answer, value_lists[last_layer_index], false);

	backward(learning_rate);

	return error;
}

float NeuralNetwork::train_batch(float** input, float** answer, const int batch_length, const int loss_function, const float learning_rate) {
	if (!network_created)
		return -1;
	
	int last_layer_size = node_list[last_layer_index];
	memset(all_gradients, 0, last_layer_size * sizeof(float));

	double error = 0;
	for (int batch_index = 0; batch_index < batch_length; batch_index++) {
		forward(input[batch_index]);

		//loss_prime
		error += loss_prime(loss_function, answer[batch_index], all_gradients, true);

	}
	error /= batch_length;

	for (int i = 0; i < last_layer_size; i++) {
		all_gradients[i] /= batch_length;
	}

	//set gradient
	memcpy(value_lists[last_layer_index], all_gradients, sizeof(float) * last_layer_size);

	backward(learning_rate);

	return error;
}

void NeuralNetwork::train_list(float** input, float** answer, const int list_length, const int loss_function, const float learning_rate, int epoches, const bool print_to_console) {
	if (!network_created)
		return;
	
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

bool NeuralNetwork::cut_network(int start_layer, int end_layer) {
	if (!network_created)
		return false;

	if (start_layer < 0 || start_layer >= node_list.size() || end_layer < 0 || end_layer >= node_list.size() || start_layer > end_layer)
		return false;


	std::vector<int> old_node_list(node_list);
	node_list.clear();
	for (int i = start_layer; i <= end_layer; i++) {
		node_list.push_back(old_node_list[i]);
	}
	std::vector<int> old_activation_function_list(activation_function_list);
	activation_function_list.clear();
	for (int i = start_layer; i < end_layer; i++) {
		activation_function_list.push_back(old_activation_function_list[i]);
	}

	last_layer_index = node_list.size() - 1;

	for (int i = 0; i < old_node_list.size(); i++) {
		delete[] original_value_lists[i];
		delete[] value_lists[i];
	}
	delete[] original_value_lists;
	delete[] value_lists;

	original_value_lists = new float* [node_list.size()];
	value_lists = new float* [node_list.size()];
	for (int i = 0; i < node_list.size(); i++) {
		original_value_lists[i] = new float[node_list[i]];
		value_lists[i] = new float[node_list[i]];
	}


	float** new_weight_lists = new float*[last_layer_index];
	float** new_bias_lists = new float* [last_layer_index];
	for (int i = 0; i < last_layer_index; i++) {
		new_weight_lists[i] = new float[node_list[i] * node_list[i+1]];
		memcpy(new_weight_lists[i], weight_lists[start_layer + i], sizeof(float) * node_list[i] * node_list[i + 1]);
		
		new_bias_lists[i] = new float[node_list[i+1]];
		memcpy(new_bias_lists[i], bias_lists[start_layer + i], sizeof(float) * node_list[i+1]);
	}


	for (int i = 0; i < old_node_list.size() - 1; i++) {
		delete[] weight_lists[i];
		delete[] bias_lists[i];
	}
	delete[] weight_lists;
	delete[] bias_lists;
	weight_lists = new_weight_lists;
	bias_lists = new_bias_lists;

	delete[] all_gradients;
	all_gradients = new float[node_list[last_layer_index]];

	return true;
}

void NeuralNetwork::save_to_file(std::string file) {

	if (!network_created)
		return;

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
	output_data[where_] = size & 0xFF;
	output_data[where_ + 1] = (size & 0xFF00) >> 8;
	output_data[where_ + 2] = (size & 0xFF0000) >> 16;
	output_data[where_ + 3] = (size & 0xFF000000) >> 24;
	where_ += length;

	//node list
	length = 4 * size;
	for (int i = 0; i < size;++i) {
		output_data[where_ + i*4] = node_list[i] & 0xFF;
		output_data[where_ + i*4 + 1] = (node_list[i] & 0xFF00) >> 8;
		output_data[where_ + i*4 + 2] = (node_list[i] & 0xFF0000) >> 16;
		output_data[where_ + i*4 + 3] = (node_list[i] & 0xFF000000) >> 24;
	}
	where_ += length;

	//activation function
	length = (size - 1) * 4;
	for (int i = 0; i < size-1; ++i) {
		output_data[where_ + i * 4] = activation_function_list[i] & 0xFF;
		output_data[where_ + i * 4 + 1] = (activation_function_list[i] & 0xFF00) >> 8;
		output_data[where_ + i * 4 + 2] = (activation_function_list[i] & 0xFF0000) >> 16;
		output_data[where_ + i * 4 + 3] = (activation_function_list[i] & 0xFF000000) >> 24;
	}
	where_ += length;

	//bias
	for (int i = 0; i < size - 1; i++) {
		length = 4 * node_list[i + 1];
		for (int n = 0; n < node_list[i + 1]; ++n) {
			int32_t u = bias_lists[i][n] * 1000000;
			output_data[where_ + n * 4] = u & 0xFF;
			output_data[where_ + n * 4 + 1] = (u & 0xFF00) >> 8;
			output_data[where_ + n * 4 + 2] = (u & 0xFF0000) >> 16;
			output_data[where_ + n * 4 + 3] = (u & 0xFF000000) >> 24;
		}
		where_ += length;
	}

	//weight
	for (int i = 0; i < size - 1; i++) {
		length = 4 * node_list[i] * node_list[i + 1];
		memcpy(&(output_data[where_]), &(weight_lists[i][0]), length);
		for (int n = 0; n < node_list[i] * node_list[i + 1]; ++n) {
			int32_t u = weight_lists[i][n] * 1000000;
			output_data[where_ + n * 4] = u & 0xFF;
			output_data[where_ + n * 4 + 1] = (u & 0xFF00) >> 8;
			output_data[where_ + n * 4 + 2] = (u & 0xFF0000) >> 16;
			output_data[where_ + n * 4 + 3] = (u & 0xFF000000) >> 24;
		}
		where_ += length;
	}

	auto myfile = std::fstream(file, std::ios::out | std::ios::binary);
	myfile.write((char*)output_data, data_size);
	myfile.close();

	delete[] output_data;
}

bool NeuralNetwork::load_from_file(std::string file) {

	// open the file
	std::ifstream file_(file, std::ios::binary);
	if (!file_.is_open()) {
		return false;
	}

	file_.unsetf(std::ios::skipws);
	std::streampos fileSize;
	file_.seekg(0, std::ios::end);
	fileSize = file_.tellg();
	file_.seekg(0, std::ios::beg);
	std::vector<uint8_t> file_data;
	file_data.reserve(fileSize);
	file_data.insert(file_data.begin(),
		std::istream_iterator<uint8_t>(file_),
		std::istream_iterator<uint8_t>());


	delete_data();

	output_error_to_file = false;

	size_t offset = 0;
	size_t length = 0;
	uint32_t u;

	//set number of layers
	int number_of_layers = 0;
	length = 4;
	number_of_layers = (((((file_data[3] << 8) + file_data[2]) << 8) + file_data[1]) << 8) + file_data[0];
	offset += length;

	last_layer_index = number_of_layers - 1;

	//create node_list
	length = 4;
	for (int i = 0; i < number_of_layers; i++) {
		int num = (((((file_data[offset + 3] << 8) + file_data[offset + 2]) << 8) + file_data[offset + 1]) << 8) + file_data[offset + 0];
		node_list.push_back(num);
		offset += length;
	}

	//set activation functions
	length = 4;
	for (int i = 0; i < number_of_layers - 1; i++) {
		int act = (((((file_data[offset + 3] << 8) + file_data[offset + 2]) << 8) + file_data[offset + 1]) << 8) + file_data[offset + 0];//*(int*)&file_data[offset]
		activation_function_list.push_back(act);
		offset += length;
	}

	//set biases
	bias_lists = new float* [number_of_layers - 1];
	for (int i = 0; i < number_of_layers - 1; i++) {
		bias_lists[i] = new float[node_list[i + 1]];
		for (int n = 0; n < node_list[i + 1]; n++) {
			uint32_t u = (((((file_data[offset + 3] << 8) + file_data[offset + 2]) << 8) + file_data[offset + 1]) << 8) + file_data[offset + 0];
			bias_lists[i][n] = (*(int32_t*) & u) / 1000000.0;
			offset += 4;
		}
	}

	//set weights
	weight_lists = new float* [number_of_layers - 1];
	for (int i = 0; i < number_of_layers - 1; i++) {
		weight_lists[i] = new float[node_list[i + 1] * node_list[i]];
		for (int n = 0; n < node_list[i + 1] * node_list[i]; n++) {
			uint32_t u = (((((file_data[offset + 3] << 8) + file_data[offset + 2]) << 8) + file_data[offset + 1]) << 8) + file_data[offset + 0];
			weight_lists[i][n] = (*(int32_t*)&u) / 1000000.0;
			offset += 4;
		}
	}

	value_lists = new float* [node_list.size()];
	original_value_lists = new float* [node_list.size()];
	for (int i = 0; i < node_list.size(); i++) {
		value_lists[i] = new float[node_list[i]];
		original_value_lists[i] = new float[node_list[i]];
	}

	all_gradients = new float[node_list[last_layer_index]];

	network_created = true;
	return true;
}
