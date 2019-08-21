/*
 * NeuralNet.cpp
 *
 *  Created on: Feb 28, 2018
 *      Author: Andy
 */

#include "NeuralNet.h"

#include <cstdlib>
#include <string>

using namespace std;

NeuralNet::NeuralNet(int inCount, int outCount, int hiddenCount, int layers) {
	this->layers = layers;
	this->inCount = inCount;
	this->outCount = outCount;
	this->hidCount = hiddenCount;
	// Sizes
	int MAX_WEIGHT = 1, MIN_WEIGHT = -1;
	int MAX_BASE = 1, MIN_BASE = -1;
	for (int i = 0; i < layers; ++i) {
		int rows = hidCount, cols = hidCount;
		if (i == 0) {
			rows = inCount;
		}
		if (i == layers - 1) {
			cols = outCount;
		}
		float* weightarr = new float[rows * cols], *basearr = new float[cols];
		for (int j = 0; j < rows * cols; ++j) {
			// Get random 0-1
			double r = (double) rand() / (RAND_MAX);
			// Scale to min/max
			weightarr[j] = (r * (MAX_WEIGHT - MIN_WEIGHT)) + MIN_WEIGHT;
		}
		for (int j = 0; j < cols; ++j) {
			// Get random 0-1
			double r = (double) rand() / (RAND_MAX);
			// Scale to min/max
			basearr[j] = 1 - (r * (MAX_BASE - MIN_BASE)) + MIN_BASE;
		}
		Matrix w(rows, cols, weightarr), b(1, cols, basearr);
		weights.push_back(w);
		bases.push_back(b);
	}

}
//Other declarator is by saveto for paralleling

Matrix NeuralNet::eval(Matrix in) {
	//Clear for use
	resetEval();
	//Copy input to  activations. Activations for input is just the inputs, no sigmoid
	activations.push_back(Matrix(in));
	for (int i = 0; i < layers; ++i) {
		// Get matrix to transform
		Matrix t = Matrix(activations.back());
		// weights dot value + bases
		t *= weights[i];
		t += bases[i];
		// Add to values
		values.push_back(t);
		// Get matrix to sigmoid
		Matrix s = Matrix(t);
		s.apply(sigmoid);
		// Add to activations
		activations.push_back(s);
	}
	//Copy output activations
	return Matrix(activations.back());
}
void NeuralNet::resetEval() {
	values.clear();
	activations.clear();
}

void NeuralNet::resetErrors() {
	weightsError.clear();
	basesError.clear();
	for (int i = 0; i < layers; ++i) {
		unsigned int r, c;
		Matrix m = weights[i];
		r = m.getr();
		c = m.getc();
		float* weightarr = new float[r * c], *basearr = new float[c];
		for (unsigned int j = 0; j < r * c; ++j) {
			weightarr[j] = 0;
		}
		for (unsigned int j = 0; j < c; ++j) {
			basearr[j] = 0;
		}
		weightsError.push_back(Matrix(r, c, weightarr));
		basesError.push_back(Matrix(1, c, basearr));
	}
}

// Single backpropogation case
void NeuralNet::backpropCase(Matrix in, Matrix expectedOut, float delta) {
	//Make case errrors matching weight and base error matricies
	vector<Matrix> caseWeightError, caseBaseError;
	for (int i = 0; i < layers; ++i) {
		int rows = hidCount, cols = hidCount;
		if (i == 0) {
			rows = inCount;
		}
		if (i == layers - 1) {
			cols = outCount;
		}
		float* weightarr = new float[rows * cols], *basearr = new float[cols];
		for (int j = 0; j < rows * cols; ++j) {
			weightarr[j] = 0;
		}
		for (int j = 0; j < cols; ++j) {
			basearr[j] = 0;
		}
		caseWeightError.push_back(Matrix(rows, cols, weightarr));
		caseBaseError.push_back(Matrix(1, cols, basearr));
	}
	//Now, begin actual backpropogation. Get the output
	Matrix output = eval(in);
	//Backprop
	for (int layer = layers - 1; layer >= 0; --layer) {
		// Get layer error by layer
		Matrix *layerError;
		// Final layer is different
		if (layer == layers - 1) {
			// Error is difference between output and expected
			layerError = new Matrix(expectedOut);
			*layerError -= output;
			// Scale by delta
			*layerError *= delta;
		} else {
			// Get base error of next layer
			// Dot product the weights of the next layer with the base error of the next layer
			weights[layer + 1].transpose();
			layerError = new Matrix(
					caseBaseError[layer + 1] * weights[layer + 1]);
			weights[layer + 1].transpose();
//			layerError->transpose();

		}

		//Finally, we have to hadamard the layer with the rate of change of that node
		Matrix rateChange = Matrix(values[layer]);
		rateChange.apply(sigmoidPrime);
		*layerError %= rateChange;
		//Now we have the layer's error.

		//The error of the base is actually equal to the layer's error
		caseBaseError[layer] = *layerError;
		//And the weight error is the previous layer's activations transposed dot the layer's error
		Matrix* weightErr = new Matrix(activations[layer]);
		weightErr->transpose();
		*weightErr *= *layerError;
		caseWeightError[layer] = *weightErr;
	}
	//Done backpropogating. Apply case to total and finish.
	for (int j = 0; j < layers; ++j) {
		basesError[j] += caseBaseError[j];
		weightsError[j] += caseWeightError[j];
	}
}
void NeuralNet::backprop(vector<Matrix> ins, vector<Matrix> outs, float delta) {
	//Evaluate and backpropogate through each and every data set.
	//Once done, alter weights and biases according to average error.

	//Catch bad cases
	if (ins.size() == 0) {
		throw invalid_argument("Backprop failure: Empty input vector.");
	}
	if (ins.size() != outs.size()) {
		throw invalid_argument(
				"Backprop failure: Mismatched input and output vectors.");
	}
	//clear error matricies
	resetErrors();

	for (unsigned int i = 0; i < ins.size(); ++i) {
		backpropCase(ins[i], outs[i], delta);
	}
	//Done evaluating. Clear then apply average errors.
	resetEval();
	for (unsigned int i = 0; i < weights.size(); ++i) {
		// Normalize for sizes
//		weightsError[i] *= (1.0 / ins.size());
		// Change
		weights[i] += weightsError[i];
	}
	for (unsigned int i = 0; i < bases.size(); ++i) {
		// Normalize for sizes
//		basesError[i] *= (1.0 / ins.size());
		bases[i] += basesError[i];
	}
}
void NeuralNet::escelate(vector<Matrix> ins, vector<Matrix> outs, int times) {
	//Trains N times
	for (int t = 0; t < times; ++t) {
		vector<Matrix> inset, outset;
		//Get 10 random test cases for each iteration that's happened, plus base 20 (escalating count)
		for (int iter = 0; iter < 20 + t * 10; ++iter) {
			//Choose a random from base set and add it
			int index = rand() % ins.size();
			inset.push_back(ins[index]);
			outset.push_back(outs[index]);
		}
		//Now, backprop set with offset
		float offset = 0.25 - (0.2 * t) / times; //Deescelating for accuracy
		backprop(inset, outset, offset);
	}
}
void NeuralNet::train(vector<Matrix> ins, vector<Matrix> outs) {
	// If size is > 100, escelate
	if (ins.size() > 100) {
		for (int k = 0; k < 10; ++k) {
			escelate(ins, outs, 10);
		}
	} else {
		// Just train off data
		for (int k = 0; k < 500; ++k) {
			backprop(ins, outs, 10);
		}
	}
}
//Load from file
NeuralNet::NeuralNet(string filename) {
	ifstream file(filename);
	for (string line; getline(file, line);) {
		Matrix a(line);
		weights.push_back(a);
		getline(file, line);
		Matrix b(line);
		bases.push_back(b);
	}
	layers = weights.size();
	Matrix m = weights[0];
	inCount = m.getr();
	hidCount = m.getc();
	Matrix e = weights[layers - 1];
	outCount = e.getc();
	file.close();

}
//Save to file
void NeuralNet::saveto(string filename) {
	ofstream file(filename);
	for (unsigned int i = 0; i < weights.size(); ++i) {
		Matrix w = weights[i];
		Matrix b = bases[i];
		file << (w.write()) << '\n' << (b.write());
		if (i + 1 < weights.size())
			file << endl;
	}
	file.close();

}

//String output
ostream &operator<<(ostream& os, NeuralNet& r) {
	os << r.toString();
	return os;
}

//String output
string NeuralNet::toString() {
	string ret = "[NET: Input Size: " + to_string(inCount);
	ret += ", Output Size: " + to_string(outCount) + ";\n";
	ret += "Layers: " + to_string(layers);
	ret += ", Width: " + to_string(hidCount) + ";\n";
	// Do matricies
	ret += "Bases:\n";
	for (unsigned int i = 0; i < bases.size(); ++i) {
		ret += to_string(i) + ": " + bases[i].toString();
	}
	ret += "\nWeights:\n";
	for (unsigned int i = 0; i < weights.size(); ++i) {
		ret += to_string(i) + ": " + weights[i].toString();
	}

	// Done
	ret += "]\n";
	return ret;
}

