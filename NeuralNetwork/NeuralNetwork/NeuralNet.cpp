#include "NeuralNet.h"


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
Neuron::Neuron(unsigned int transferFunctionUsed, const Layer& nextLayer)
{
	for (size_t i = 0; i < nextLayer.size(); ++i) {

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

double Neuron::transferFunction(double x)
{
	return (possibleFunctions->*transfer)(x);
}

double Neuron::transferFunctionDerivative(double x)
{
	return (possibleFunctions->*derivative)(x);
}