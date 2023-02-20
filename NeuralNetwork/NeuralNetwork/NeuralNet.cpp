#include "NeuralNet.h"

Connection::Connection(double weight, double deltaWeight)
{
	this->weight = weight;
	this->deltaWeight = deltaWeight;
}

//###########################################################Neuron Transfer Functions#################################################
double NeuronTransferFunctions::hyperbolicTangent(double x)
{
	return tanh(x);
}
double NeuronTransferFunctions::hyperbolicTangentDerivative(double x)
{
	double result = 1 - tanh(x) * tanh(x);
	return result;
}
double NeuronTransferFunctions::sigmoid(double x)
{
	double result = 1 / (1 + exp(-x));
	return result;
}
double NeuronTransferFunctions::sigmoid_derivative(double x)
{
	double result = sigmoid(x) * (1 - sigmoid(x));
	return result;
}
double NeuronTransferFunctions::ReLU_Function(double x)
{
	if (x > 0)
		return x;
	return 0;
}
double NeuronTransferFunctions::ReLU_Derivative(double x)
{
	if (x > 0)
		return 1;
	return 0;
}


//###########################################################Neuron Class##############################################################
Neuron::Neuron(unsigned int transferFunctionUsed, size_t number_outputs)
{
	int range_from = -1000;
	int range_to = 1000;
	std::random_device                  rand_dev;
	std::mt19937                        generator(rand_dev());
	std::uniform_int_distribution<int>  distr(range_from, range_to);

	double randomWeight = (double) distr(generator) / 100;

	outConnections.clear();
	for (size_t i = 0; i < number_outputs; ++i) {
		outConnections.push_back(Connection(randomWeight, 0.0));
	}

	switch (transferFunctionUsed) {
	case 0:
		transfer = &NeuronTransferFunctions::hyperbolicTangent;
		derivative = &NeuronTransferFunctions::hyperbolicTangentDerivative;

	case 1:
		transfer = &NeuronTransferFunctions::ReLU_Function;
		derivative = &NeuronTransferFunctions::ReLU_Derivative;

	case 2:
		transfer = &NeuronTransferFunctions::sigmoid;
		derivative = &NeuronTransferFunctions::sigmoid_derivative;
	}
}
Neuron::~Neuron()
{

}

double Neuron::transferFunction(double x)
{
	return (possibleFunctions->*transfer)(x);
}

double Neuron::transferFunctionDerivative(double x)
{
	return (possibleFunctions->*derivative)(x);
}