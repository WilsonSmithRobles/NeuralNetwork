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
Neuron::Neuron(unsigned int transferFunctionUsed, size_t number_outputs, unsigned int my_index)
{
	int range_from = -1000;
	int range_to = 1000;
	std::random_device                  rand_dev;
	std::mt19937                        generator(rand_dev());
	std::uniform_int_distribution<int>  distr(range_from, range_to);

	double randomWeight = (double) distr(generator) / 1000;

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

	this->eta = 0.5;
	this->alpha = 0.3;
	this->myIndex = my_index;
}
Neuron::~Neuron()
{}

void Neuron::feedForward(const Layer& prevLayer)
{
	double sum = 0;
	for (size_t neuron = 0; neuron < prevLayer.size(); ++neuron) {
		sum += prevLayer[neuron].getOutput() * 
			prevLayer[neuron].outConnections[this->myIndex].weight;
	}
}

void Neuron::updateInputWeights(Layer& prevLayer)
{
	// The weights to be updated are in the Connection container
	// in the neurons in the preceeding layer.

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.outConnections[this->myIndex].deltaWeight;			//Get the weight prior to updating

		double newDeltaWeight =
			//Individual input, magnified by the gradient and train rate.
			//eta = overall net learning rate -> 0.0 - slow learner; 1.0 - reckless learner
			eta
			* neuron.getOutput()
			* this->myGradient
			//Also add momentum = a fraction of the previous delta_weight
			//alpha = momentum -> 0.0 - no momentum; 0.5 - moderate momentum.
			+ this->alpha
			* oldDeltaWeight
			;

		neuron.outConnections[this->myIndex].deltaWeight = newDeltaWeight;
		neuron.outConnections[this->myIndex].weight += newDeltaWeight;
	}
};

double Neuron::myErrorContrib(const Layer& nextLayer) const
{
	double sum = 0.0;

	//Sum our contributions of the errors at the nodes we feed.

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
		sum += this->outConnections[n].weight * nextLayer[n].m_gradient;			//For each weight to the right we multiply by gradient of that neuron ignoring the bias neuron to the right and add it to sum.
	}
	return sum;
};

void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
	double dow = this->myErrorContrib(nextLayer);
	this->myGradient = dow * Neuron::transferFunctionDerivative(this->my_output);
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - this->my_output;
	this->myGradient = delta * Neuron::transferFunctionDerivative(this->my_output);
};

double Neuron::transferFunction(double x)
{
	return (possibleFunctions->*transfer)(x);
}
double Neuron::transferFunctionDerivative(double x)
{
	return (possibleFunctions->*derivative)(x);
}



//###########################################################Net Class##############################################################
NeuralNet::NeuralNet(void)
{}
NeuralNet::~NeuralNet(void)
{}
NeuralNet::NeuralNet(const std::vector<unsigned int> topology, const std::vector<unsigned int> transferFunctions)
{
	size_t numLayers = topology.size();
	for (size_t layerNum = 0; layerNum < numLayers; ++layerNum) {
		myLayers.push_back(Layer());
		unsigned int numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
		for (size_t i = 0; i <= topology[layerNum]; ++i) {
			myLayers[layerNum].push_back(Neuron(transferFunctions[layerNum], numOutputs, numOutputs));
		}

		//Force the bias node's output value to 1.0. It's the last neuron created above.
		myLayers.back().back().setOutputVal(1.0);
	}

	this->m_error = 0.0;
	this->m_recentAverageError = 0.0;
	this->m_recentAverageSmoothingFactor = 0.0;
}

void NeuralNet::setTopology(const std::vector<unsigned int> topology, const std::vector<unsigned int> transferFunctions)
{
	this->resetNet();

	size_t numLayers = topology.size();
	for (size_t layerNum = 0; layerNum < numLayers; ++layerNum) {
		myLayers.push_back(Layer());
		unsigned int numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
		for (size_t i = 0; i <= topology[layerNum]; ++i) {
			myLayers[layerNum].push_back(Neuron(transferFunctions[layerNum], numOutputs, numOutputs));
		}

		//Force the bias node's output value to 1.0. It's the last neuron created above.
		myLayers.back().back().setOutputVal(1.0);
	}

	this->m_error = 0.0;
	this->m_recentAverageError = 0.0;
	this->m_recentAverageSmoothingFactor = 0.0;
}
void NeuralNet::resetNet()
{
	this->myLayers.clear();
	this->m_error = 0;
	this->m_recentAverageError = 0;
	this->m_recentAverageSmoothingFactor = 0;
}

void NeuralNet::feedForward(const std::vector<double> Inputs)
{
	assert(Inputs.size() == myLayers[0].size() - 1);

	// Assign (latch) the input values into the input neurons
	for (unsigned inputNeuron = 0; inputNeuron < Inputs.size(); ++inputNeuron) {
		myLayers[0][inputNeuron].setOutputVal(Inputs[inputNeuron]);
	}

	// Forward propagate
	for (unsigned layerNum = 1; layerNum < myLayers.size(); ++layerNum) {
		Layer& prevLayer = myLayers[layerNum - 1];
		for (unsigned n = 0; n < myLayers[layerNum].size() - 1; ++n) {
			myLayers[layerNum][n].feedForward(prevLayer);
		}
	}
}
void NeuralNet::backPropagation(const std::vector<double> targets)
{

}
unsigned NeuralNet::getMaximizedOutput() 
{
	Layer& outputLayer = this->myLayers.back();
	size_t numOutputs = this->myLayers.back().size() - 1;

	std::vector<double> outputs;

	for (size_t outNeuron = 0; outNeuron < numOutputs; ++outNeuron) {
		outputs.push_back(outputLayer[outNeuron].getOutputVal());
	}

	auto max_it = std::max_element(outputs.begin(), outputs.end());
	size_t index = std::distance(outputs.begin(), max_it);

	return index;
}
std::vector<double> NeuralNet::getResults()
{
	Layer& outputLayer = this->myLayers.back();
	size_t numOutputs = this->myLayers.back().size() - 1;

	std::vector<double> outputs;

	for (size_t outNeuron = 0; outNeuron < numOutputs; ++outNeuron) {
		outputs.push_back(outputLayer[outNeuron].getOutputVal());
	}

	return outputs;
}

void NeuralNet::updateNetEtas(double newEtas)
{
	for (unsigned i = 0; i < this->myLayers.size() - 1; ++i) {
		for (unsigned j = 0; j < this->myLayers[i].size(); ++j) {
			this->myLayers[i][j].updateEta(newEtas);
		}
	}
}