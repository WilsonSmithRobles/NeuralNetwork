#include "filesReaderWriter.h"
size_t filesReaderWriter::countFileLines(std::string fileName) 
{
	std::ifstream file(fileName);
	if (!file.is_open()) {
		std::cerr << "Error: Unable to open file " << fileName << std::endl;
		return 0;
	}

	size_t lineCount = 0;
	std::string line;
	while (std::getline(file, line)) {
		++lineCount;
	}

	file.close();

	return lineCount;
}
std::vector<std::vector<double>> filesReaderWriter::readCSV(std::string csv_filename) 
{
	std::ifstream readingFile;
	readingFile.open(csv_filename);
	std::string line;

	std::vector<std::vector<double>> values_read;

	if (!readingFile.is_open()) {
		std::cout << "readCSV:: Could not open File";
		return values_read;
	}
		
	while (std::getline(readingFile, line)) {
		std::vector<double> row_values;
		std::ifstream full_line(line);
		std::string value;
		while (std::getline(full_line, value, ',')) {
			row_values.push_back(std::stod(value));
		}
		values_read.push_back(row_values);
	}

	readingFile.close();
	return values_read;
}
void filesReaderWriter::writeCSV(std::vector<std::vector<double>> vector_to_write, std::string fileName)
{
	if (fileExists(fileName)) {
		std::cout << "File already exists" << std::endl;
		return;
	}

	std::ofstream writefile;
	writefile.open(fileName);

	size_t vectorSize = vector_to_write.size();
	size_t columnSize;
	for (size_t vector_row = 0; vector_row < vectorSize; ++vector_row) {
		columnSize = vector_to_write[vector_row].size() - 1;
		for (size_t vector_col = 0; vector_col < columnSize; ++vector_col) {
			writefile << vector_to_write[vector_row][vector_col] << ",";
		}
		writefile << vector_to_write[vector_row][columnSize - 1] << std::endl;
	}

	writefile.close();
}
bool filesReaderWriter::fileExists(const std::string& filename)
{
	struct stat buf;
	if (stat(filename.c_str(), &buf) != -1)
	{
		return true;
	}
	return false;
}