/*
 * NeuralNet.h
 *
 *  Created on: Feb 28, 2018
 *      Author: Andy
 */

#ifndef NEURALNET_H_
#define NEURALNET_H_
using namespace std;
#include "../math/Matrix.h"
#include "../math/sigFunctions.h"
#include <vector>
#include <stdexcept>
#include <sstream>
#include <fstream>
class NeuralNet {
private:
	int layers;
	int inCount, outCount, hidCount;
	// Net constants
	vector<Matrix> weights;
	vector<Matrix> bases;
	// Evaluations
	vector<Matrix> values;
	vector<Matrix> activations;
	// Clear evaluations
	void resetEval();
	// Learning
	vector<Matrix> weightsError;
	vector<Matrix> basesError;
	// Clear errors
	void resetErrors();

	// Single back propogation
	void backpropCase(Matrix in, Matrix out, double);
	//Training method of selection sampling
	void batchTrain(vector<Matrix>, vector<Matrix>, int, double);
public:
	//Declare randomized/ "new" neural net
	NeuralNet(int inCount, int outCount, int hiddenCount, int layers);
	//Load from file
	NeuralNet(string);
	//Evaluate input
	Matrix eval(Matrix);
	//Generic "Train". Set to whatever the developer thinks fits
	void train(vector<Matrix>, vector<Matrix>);
	//Backpropogation is the base learning method
	void backprop(vector<Matrix>, vector<Matrix>, double);

	//Readable output
	std::string toString();
	//Save neural net's current state to file
	void saveto(string);
};
//Output
ostream &operator<<(ostream&, NeuralNet&);

#endif /* NEURALNET_H_ */
