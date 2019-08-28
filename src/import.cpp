/*
 * import.cpp
 *
 *  Created on: Feb 28, 2018
 *      Author: Andy
 */

#include "import.h"

using namespace std;

FileReader::FileReader(string filename) {
	// Read file
	ifstream file(filename);
	if (!file.is_open()) {
		throw runtime_error("File at " + filename + "not found");
	}
	// Read each line
	for (string line; getline(file, line);) {
		// Parse line
		istringstream read(line);
		long double *iarr = new long double[2];
		long double *oarr = new long double[3];
		// Load data
		read >> iarr[0] >> iarr[1] >> oarr[0] >> oarr[1] >> oarr[2];
		// Push to vectors
		inputs.push_back(Matrix(1, 2, iarr));
		outputs.push_back(Matrix(1, 3, oarr));
	}
	file.close();
}

vector<Matrix> FileReader::getInputs() {
	return inputs;
}
vector<Matrix> FileReader::getOutputs() {
	return outputs;
}
