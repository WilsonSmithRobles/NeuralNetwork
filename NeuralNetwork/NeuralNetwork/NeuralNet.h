#pragma once

#include <vector>
#include <math.h>

struct NeuronTransferFunctions
{
	double hyperbolicTangent(double x);
	double hyperbolicTangentDerivative(double x);
	double sigmoid(double x);
	double sigmoid_derivative(double x);
	double ReLU_Function(double x);
	double ReLU_Derivative(double x);

};


struct Connection
{
	double weight, deltaWeight;
};
typedef std::vector<Neuron> Layer;

class Neuron {
public:
	Neuron(size_t transferFunctionUsed);
	~Neuron();
	double get_output();


private:
	double transferFunction(double Input);
	double transferFunctionDerivative(double x);
	double my_output;
	std::vector<Connection> outConnections;

};

class NeuralNet
{
public:
	NeuralNet();
	~NeuralNet();
	void feedForward(std::vector<double> Inputs);
	void backPropagation(std::vector<double> Targets);
	int getMaximizedOutput();

private:

};

