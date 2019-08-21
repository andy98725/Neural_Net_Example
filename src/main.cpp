/*	Improved Neural Net by Andy Hudson
 *	Differences from BasicNeuralNet:
 *	- Used vector storage instead of highly bugged array storage
 *	- Garbage collection functional
 *	- All around efficiency improvements
 *	- Actually runs
 *	Usage:
 *	1. Load training data from file into vector of input and output matricies
 *		(May have to hard code how to read it, or create it in your own way)
 *	2. Initiate neural net class with input count, output count, hidden count/layer, and total layer count (including input)
 *	3. Train neural net with matrix vectors
 *	4. Evaluate with input matrixW
 */

#include <iostream>
#include <vector>

#include "import.h"
#include "math/Matrix.h"
#include "NN/NeuralNet.h"

using namespace std;

// Test method
void xorTest();
// basic NN training method
void basicTrain();

// Use data
void trainFromData(NeuralNet&);
int main() {
	// Test funct
	xorTest();
	cout << "Ending..." << endl;
	cout.flush();
	return 0;
}

void xorTest() {
	// Dummy data
	Matrix a(1, 2, new float[2] { 0, 0 });
	Matrix b(1, 2, new float[2] { 0, 1 });
	Matrix c(1, 2, new float[2] { 1, 0 });
	Matrix d(1, 2, new float[2] { 1, 1 });
	Matrix yes(1,1,new float[1]{1});
	Matrix no(1,1,new float[1]{0});

	// Init basic net
	NeuralNet net(2, 1, 4, 2); // @suppress("Ambiguous problem")
	// Initial eval
	Matrix preA = net.eval(a), preB = net.eval(b), preC = net.eval(c), preD =
			net.eval(d);
	cout << "Initial: " << endl;
	// cout << "Net: " << net << endl;
	cout << "0 0: " << preA;
	cout << "0 1: " << preB;
	cout << "1 0: " << preC;
	cout << "1 1: " << preD;
	// Update
	cout.flush();

	// Train with dummy data
	vector<Matrix> ins, outs;
	ins.push_back(a);
	outs.push_back(no);
//	ins.push_back(b); TODO: Restore full training data
//	outs.push_back(yes);
//	ins.push_back(c);
//	outs.push_back(yes);
	ins.push_back(d);
	outs.push_back(no);
	net.train(ins, outs);



	// Result eval
	Matrix resA = net.eval(a), resB = net.eval(b), resC = net.eval(c), resD =
			net.eval(d);
	cout << "Result: " << endl;
	// cout << "Net: " << net << endl;
	cout << "0 0: " << resA;
	cout << "0 1: " << resB;
	cout << "1 0: " << resC;
	cout << "1 1: " << resD;

}

void basicTrain() {
	//Load NN from file
	NeuralNet baby("NNets/beginnerNet.nn");

	cout << "Initiated net.\n";
	//Sample evaluation
	Matrix out = baby.eval(Matrix { { 0, 0 } });
	cout << "Pre eval at 0,0:\n" << out;
	cout << "Training net...\n";
	//Training function
	trainFromData(baby);
	//Sample evaluation
	out = baby.eval(Matrix { { 0, 0 } });
	cout << "Post eval at 0,0:\n" << out;
	//Save work
	baby.saveto("NNets/beginnerNet.nn");
	//Generate image
	//FileWriter ex(469,376, baby, "out.txt");
}
void trainFromData(NeuralNet &net) {
	//Read data from file
	FileReader import("training.txt");
	cout << "Loaded training data.\nBeginning training...\n";
	//Train network
	net.train(import.getInputs(), import.getOutputs());
	cout << "Training complete.\n";
}
