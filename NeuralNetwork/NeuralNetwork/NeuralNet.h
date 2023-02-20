#pragma once
#include <vector>
#include <math.h>
#include <random>

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
	Connection(double weight, double deltaWeight);
	double weight, deltaWeight;
};
typedef std::vector<Neuron> Layer;

class Neuron {
public:
	Neuron(unsigned int transferFunctionUsed, size_t number_outputs);
	~Neuron();

	double get_output();
	void feedForward(const Layer& prevLayer);

	//Properties
	std::vector<Connection> outConnections;

private:
	double transferFunction(double Input);
	double transferFunctionDerivative(double x);
	double my_output;

	double(NeuronTransferFunctions::* transfer)(double x);
	double(NeuronTransferFunctions::* derivative)(double x);
	NeuronTransferFunctions* possibleFunctions;

	//Properties
	double eta;
	double alpha;
};



class NeuralNet
{
public:
	NeuralNet();
	NeuralNet(std::vector<unsigned int> topology, std::vector<unsigned int> transferFunctions);
	~NeuralNet();
	void setTopology(std::vector<unsigned int> topology, std::vector<unsigned int> transferFunctions);
	void resetNet();
	void feedForward(std::vector<double> Inputs);
	void backPropagation(std::vector<double> Targets);
	int getMaximizedOutput();

private:
	//Properties
	std::vector<Layer> myLayers;
};

