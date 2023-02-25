#include "TrainingData.h"


TrainingData::TrainingData(std::string csv_samplesFile, std::string csv_topology_and_functions)
{
	std::vector<std::vector<double>> topology_functions = this->fileManager.readCSV(csv_topology_and_functions);
	size_t topologyFileSize = topology_functions.size();
	if (topologyFileSize != 2){
		std::cout << "Bad topology_functions file" << std::endl;
		return;
	}

	size_t topology_layers = topology_functions[0].size();
	for (size_t i = 0; i < topology_layers; ++i) {
		this->topology.emplace_back((unsigned)topology_functions[0][i]);
	}

	size_t transferFunctions = topology_functions[1].size();
	this->transferFunctions.emplace_back((unsigned) 1);
	for (size_t i = 0; i < transferFunctions; ++i) {
		this->transferFunctions.emplace_back((unsigned)topology_functions[1][i]);
	}

	std::vector<std::vector<double>> csv_values = this->fileManager.readCSV(csv_samplesFile);
	std::vector<double> input;
	std::vector<double> target;
	std::vector<double> target_reference;
	unsigned outputsNeuralNet = this->topology.back();
	for (unsigned i = 0; i < outputsNeuralNet; ++i) {
		target_reference.emplace_back(-1.0);
	}

	//Iteramos sobre todos los valores en el csv. A partir de estos creamos los dos vectores de vectores inputs y targets. 1/fila.
	//Creamos también un vector de índices para aleatorizar los grupos luego.
	size_t rowsSize = csv_values.size();
	std::vector<size_t> index_vector(rowsSize);
	size_t colsSize;
	for (size_t row = 0; row < rowsSize; ++row) {
		colsSize = csv_values[row].size();
		for (size_t col = 1; col < colsSize; ++col){
			input.push_back(csv_values[row][col]);
		}
		target = target_reference;
		target[(unsigned)csv_values[row][0]] = 1.0;

		this->allSamples.inputs.push_back(input);
		this->allSamples.targets.push_back(target);

		index_vector[row] = row;

		input.clear();
		target.clear();
	}

	//Una vez tenemos todos los valores del csv almacenados en orden en la clase hay que aleatorizar los diferentes vectores de 
	//autoeval, train, validacion.
	std::random_shuffle(index_vector.begin(), index_vector.end());

	for (size_t i = 0; i < rowsSize; ++i) {
		if (i < rowsSize / 2) {
			this->Training.inputs.push_back(this->allSamples.inputs[index_vector[i]]);
			this->Training.targets.push_back(this->allSamples.targets[index_vector[i]]);
			continue;
		}
		if (i < rowsSize * 2 / 3) {
			this->Autoevaluation.inputs.push_back(this->allSamples.inputs[index_vector[i]]);
			this->Autoevaluation.targets.push_back(this->allSamples.targets[index_vector[i]]);
			continue;
		}
		this->Validation.inputs.push_back(this->allSamples.inputs[index_vector[i]]);
		this->Validation.targets.push_back(this->allSamples.targets[index_vector[i]]);
	}
}
TrainingData::~TrainingData(void)
{}

LabeledInputs TrainingData::getShuffledTrainingVectors(void)
{
	return this->shuffleLabeledInputs(this->Training);
}
LabeledInputs TrainingData::getShuffledAutoevalVectors(void)
{
	return this->shuffleLabeledInputs(this->Training);
}
LabeledInputs TrainingData::getShuffledValidationVectors(void)
{
	return this->shuffleLabeledInputs(this->Training);
}

LabeledInputs TrainingData::shuffleLabeledInputs(LabeledInputs tags2shuffle)
{
	size_t labelsSize = tags2shuffle.inputs.size();
	std::vector<unsigned> index_vector(labelsSize);
	LabeledInputs shuffledLabels;

	for (unsigned i = 0; i < labelsSize; ++i) {
		index_vector[i] = i;
	}

	std::random_shuffle(index_vector.begin(), index_vector.end());

	for (size_t i = 0; i < labelsSize; ++i) {

	}

}