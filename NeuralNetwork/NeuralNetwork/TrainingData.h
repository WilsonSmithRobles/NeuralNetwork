#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include "filesReaderWriter.h"

struct LabeledInputs
{
	std::vector<std::vector<double>> inputs;
	std::vector<std::vector<double>> targets;
};

class TrainingData
{
	TrainingData(std::string csv_samplesFile, std::string csv_topology_and_functions);
	~TrainingData(void);

public:
	std::vector<unsigned> topology;
	std::vector<unsigned> transferFunctions;

	LabeledInputs getShuffledTrainingVectors(void);
	LabeledInputs getShuffledAutoevalVectors(void);
	LabeledInputs getShuffledValidationVectors(void);

private:
	filesReaderWriter fileManager;
	LabeledInputs allSamples;
	LabeledInputs Training;
	LabeledInputs Autoevaluation;
	LabeledInputs Validation;
	LabeledInputs shuffleLabeledInputs(const LabeledInputs& tags2shuffle);
};

