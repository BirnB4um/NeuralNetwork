#include <iostream>
#include "NeuralNetwork.h"
#include <fstream>
#include <string>

#include <SFML/Graphics.hpp>

template<class T>
void print(const T c) {
	std::cout << c << std::endl;
}

unsigned int MAX_ITER = 200;

int iterate(double real, double complex, int max_iter) {
	double zReal = real;
	double zComplex = complex;
	double r2, c2;
	for (int iter = 1; iter < max_iter; iter++) {
		r2 = zReal * zReal;
		c2 = zComplex * zComplex;

		zComplex = 2.0 * zReal * zComplex + complex;
		zReal = r2 - c2 + real;

		if (zReal * zReal + zComplex * zComplex > 4.0) {
			return iter;
		}
	}
	return max_iter;
}

unsigned char* read_mnist_labels(std::string full_path, int& number_of_labels) {
	auto reverseInt = [](int i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	typedef unsigned char uchar;

	std::ifstream file(full_path, std::ios::binary);

	if (file.is_open()) {
		int magic_number = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

		file.read((char*)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

		uchar* _dataset = new uchar[number_of_labels];
		for (int i = 0; i < number_of_labels; i++) {
			file.read((char*)&_dataset[i], 1);
		}
		return _dataset;
	}
	else {
		throw std::runtime_error("Unable to open file `" + full_path + "`!");
	}
}

unsigned char** read_mnist_images(std::string full_path, int& number_of_images, int& image_size) {
	auto reverseInt = [](int i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	typedef unsigned char uchar;

	std::ifstream file(full_path, std::ios::binary);

	if (file.is_open()) {
		int magic_number = 0, n_rows = 0, n_cols = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

		file.read((char*)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

		image_size = n_rows * n_cols;

		uchar** _dataset = new uchar * [number_of_images];
		for (int i = 0; i < number_of_images; i++) {
			_dataset[i] = new uchar[image_size];
			file.read((char*)_dataset[i], image_size);
		}
		return _dataset;
	}
	else {
		throw std::runtime_error("Cannot open file `" + full_path + "`!");
	}
}

int main() {
	//int number_lables;
	//unsigned char* lable_data;
	//lable_data = read_mnist_labels("t10k-labels-idx1-ubyte.dat", number_lables);

	//int number_images;
	//int images_size;
	//unsigned char** image_data;
	//image_data = read_mnist_images("t10k-images-idx3-ubyte.dat", number_images, images_size);

	float width = 3;
	float height = 3;
	float off_x = -2.1;
	float off_y = 1.5;

	if (true) {
		NeuralNetwork _nn;
		_nn.load_from_file("test_save.dat");

		sf::Image img;
		std::cout << "save image..." << "\n";
		img.create(400, 400);
		float* in = new float[2];
		for (int y = 0; y < 400; y++) {
			for (int x = 0; x < 400; x++) {
				in[0] = off_x + x * width / 400;
				in[1] = off_y - y * height / 400;
				int pixel = _nn.forward(in)[0] * 255;
				pixel = pixel < 0 ? 0 : pixel > 255 ? 255 : pixel;
				img.setPixel(x, y, sf::Color(pixel, pixel, pixel, 255));
			}
		}
		img.saveToFile("load_ai_brot.png");
		std::cout << "saved image" << "\n";

		return 0;
	}

	//sf::Image i;
	//i.create(1000, 1000);

	float** input;
	float** answer;
	input = new float* [1000 * 1000];
	answer = new float* [1000 * 1000];

	float X, Y;
	int i = 0;
	for (int y = 0; y < 1000; y++) {
		for (int x = 0; x < 1000; x++) {
			input[i] = new float[2];
			answer[i] = new float[1];

			X = off_x + x * width / 1000;
			Y = off_y - y * height / 1000;

			input[i][0] = X;
			input[i][1] = Y;
			float iter = float(iterate(X, Y, 255)) / 255;
			answer[i][0] = iter;

			//pixel = pixel == 255 ? 0 : pixel;
			//i.setPixel(x, y, sf::Color(pixel, pixel, pixel, 255));
			i++;
		}
	}
	//i.saveToFile("mandelbrot.png");

	int* new_index = new int[1000 * 1000];
	for (int i = 0; i < 1000 * 1000; i++) {
		new_index[i] = i;
	}
	std::random_shuffle(new_index, new_index + 1000000);

	//CREATE
	srand(time(NULL));
	NeuralNetwork nn;
	nn.add_layer(2, TANH);
	nn.add_layer(200, TANH);
	nn.add_layer(200, TANH);
	nn.add_layer(200, TANH);
	nn.add_layer(200, SIGMOID);
	nn.add_layer(1);
	nn.create();
	nn.set_output_error_to_file(true, "error.txt");

	//TRAIN
	std::ofstream file;
	file.open("error.txt");

	float start_learning_rate = 0.01;
	float end_learning_rate = 0.001;
	int epoche_count = 1;

	float learning_rate = 0.01f;
	float  error = 0;

	sf::Image img;
	img.create(100, 100);
	float* in = new float[2];
	int c = 0;
	for (int epoche = 0; epoche < epoche_count; epoche++) {
		//learning_rate = start_learning_rate + epoche * (end_learning_rate - start_learning_rate) / epoche_count;
		learning_rate = 0.001f;

		for (int index = 0; index < 1000 * 20; index++) {
			//i = rand() % 1000000;
			i = new_index[index];
			//i = (rand() % 1000) * 1000 + (rand() % 1000);

			error = nn.train_once(input[i], answer[i], MSE, learning_rate);

			if (index % 1000 == 0) {
				std::cout << epoche << "  " << index << " " << error << "\n";
				file << error << "\n";
			}

			//if (index % 20000 == 0) {
			//	std::cout << "save image..." << "\n";
			//	for (int y = 0; y < 100; y++) {
			//		for (int x = 0; x < 100; x++) {
			//			in[0] = off_x + x * width / 100;
			//			in[1] = off_y - y * height / 100;
			//			int pixel = nn.forward(in)[0] * 255;
			//			pixel = pixel < 0 ? 0 : pixel > 255 ? 255 : pixel;
			//			img.setPixel(x, y, sf::Color(pixel, pixel, pixel, 255));
			//		}
			//	}
			//	img.saveToFile("ai_output/ai_brot" + std::to_string(c) + ".png");
			//	c++;
			//	std::cout << "saved image" << "\n";
			//}
		}
	}
	file.close();

	//sf::Image img;
	std::cout << "save image..." << "\n";
	img.create(100, 100);
	//float* in = new float[2];
	for (int y = 0; y < 100; y++) {
		for (int x = 0; x < 100; x++) {
			in[0] = off_x + x * width / 100;
			in[1] = off_y - y * height / 100;
			int pixel = nn.forward(in)[0] * 255;
			pixel = pixel < 0 ? 0 : pixel;
			img.setPixel(x, y, sf::Color(pixel, pixel, pixel, 255));
		}
	}
	img.saveToFile("ai_brot.png");
	std::cout << "saved image" << "\n";

	nn.save_to_file("test_save.dat");

	system("py err_plot.py");

	delete[] input;
	delete[] answer;
	return 0;
}