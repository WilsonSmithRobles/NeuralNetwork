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
	Neuron(unsigned int transferFunctionUsed, size_t number_outputs, unsigned int my_index);
	~Neuron(void);

	void setOutput(double output) { this->my_output = output; };
	double getOutput() { return my_output; };

	void feedForward(const Layer& prevLayer);

	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer);

	std::vector<Connection> getOutputWeights() { return outConnections; }
	void setOutputWeights(std::vector<Connection> newWeights) { this->outConnections = newWeights; }
	void updateEta(double newEta) { this->eta = newEta; }

private:
	double transferFunction(double Input);
	double transferFunctionDerivative(double x);
	double my_output;

	double(NeuronTransferFunctions::* transfer)(double x);
	double(NeuronTransferFunctions::* derivative)(double x);
	NeuronTransferFunctions* possibleFunctions;

	double myErrorContrib(const Layer& nextLayer) const;

	//Properties
	double eta;
	double alpha;
	std::vector<Connection> outConnections;
	unsigned myIndex;
	double myGradient;
};



class NeuralNet
{
public:
	NeuralNet(void);
	NeuralNet(const std::vector<unsigned int> topology, const std::vector<unsigned int> transferFunctions);
	~NeuralNet();

	void setTopology(const std::vector<unsigned int> topology, const std::vector<unsigned int> transferFunctions);
	void resetNet();

	//Basic operations from a NeuralNet.
	void feedForward(const std::vector<double> Inputs);
	void backPropagation(const std::vector<double> Targets);
	int getMaximizedOutput();
	std::vector<double> getResults();

	//Errors are only computed while training.
	double getRecentAverageError(void) const { return m_recentAverageError; }
	double getLastError(void) const { return m_error; }

private:
	std::vector<Layer> myLayers;
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;
	void updateNetEtas(double newEtas);
};

