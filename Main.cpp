#include <iostream>
#include "NeuralNetwork.h"

template<class T>
void print(const T c) {
	std::cout << c << std::endl;
}

int main() {
	srand(time(NULL));

	NeuralNetwork nn;
	nn.add_layer(2, TANH);
	nn.add_layer(5, TANH);
	nn.add_layer(5, TANH);
	nn.add_layer(5, TANH);
	nn.add_layer(5, TANH);
	nn.add_layer(2);//output-layer doesnt need activationfunction

	nn.create();

	float in[4][2];
	float out[4][2];
	in[0][0] = 0;
	in[0][1] = 0;
	out[0][0] = 1;
	out[0][1] = 0;

	in[1][0] = 0;
	in[1][1] = 1;
	out[1][0] = 1;
	out[1][1] = 1;

	in[2][0] = 1;
	in[2][1] = 0;
	out[2][0] = 0;
	out[2][1] = 1;

	in[3][0] = 1;
	in[3][1] = 1;
	out[3][0] = 0;
	out[3][1] = 0;

	for (int n = 0; n < 500; n++) {
		for (int i = 0; i < 4; i++) {
			nn.train(in[i], out[i], MSE, 0.01);
		}
		std::cout << n << "\n";
	}

	std::cout << nn.forward(in[0])[0] << " " << nn.forward(in[0])[1] << std::endl;
	std::cout << nn.forward(in[1])[0] << " " << nn.forward(in[1])[1] << std::endl;
	std::cout << nn.forward(in[2])[0] << " " << nn.forward(in[2])[1] << std::endl;
	std::cout << nn.forward(in[3])[0] << " " << nn.forward(in[3])[1] << std::endl;

	std::cout << "end";
	std::cin.get();
	return 0;
}