#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>


class filesReaderWriter
{
public:
	size_t countFileLines(std::string fileName);
	std::vector<std::vector<double>> readCSV(std::string csv_filename);
	void writeCSV(std::vector<std::vector<double>> vector_to_write, std::string fileName);

private:
	//Check if a file exists
	bool fileExists(const std::string& filename);	//@return true if the file exists, else false
};

