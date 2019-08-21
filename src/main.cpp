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
#include <string>
#include <vector>

#include "export.h"
#include "import.h"
#include "math/Matrix.h"
#include "NN/NeuralNet.h"

using namespace std;

// Image sizes
#define IMG_WID 469
#define IMG_HEI 376

// Test method
void xorTest();
// basic NN training methods
void startTrain(string fileloc, int layers, int hiddenWid);
void continueTrain(string fileloc, string filedest, string imgdest);

// Use data
void trainFromData(NeuralNet&);
int main() {
	// Basic train funct
//	startTrain("NNets/sampleNet.nn", 8, 20);
	continueTrain("NNets/sampleNet.nn", "NNets/continueNet.nn", "output/out2.txt");
//	// Test funct
//	xorTest();
	cout << "Ending..." << endl;
	cout.flush();
	return 0;
}

void xorTest() {
	// Dummy data
	Matrix a(1, 2, new double[2] { 0, 0 });
	Matrix b(1, 2, new double[2] { 0, 1 });
	Matrix c(1, 2, new double[2] { 1, 0 });
	Matrix d(1, 2, new double[2] { 1, 1 });
	Matrix yes(1, 1, new double[1] { 1 });
	Matrix no(1, 1, new double[1] { 0 });

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
	ins.push_back(d);
	outs.push_back(no);

	ins.push_back(b);
	outs.push_back(yes);
	ins.push_back(c);
	outs.push_back(yes);

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

void startTrain(string fileloc, int layers, int hiddenWid) {
	// Make initial NN
	NeuralNet net(2, 3, hiddenWid, layers);	// @suppress("Ambiguous problem")
	cout << "Initiated net." << endl << "Making pre-image..." << endl;
	cout.flush();

	// Make image
	FileWriter pre(IMG_WID, IMG_HEI, net, "preOut.txt");

	// Train net
	cout << "Image complete. Training net...\n";
	cout.flush();
	//Training function
	trainFromData(net);

	//Save work
	cout << "Saving net and making image..." << endl;
	cout.flush();
	net.saveto(fileloc);
	//Generate image
	FileWriter ex(IMG_WID, IMG_HEI, net, "out.txt");

}

void continueTrain(string fileloc, string filedest, string imgdest) {
	//Load NN from file
	NeuralNet net(fileloc);
	cout << "Initiated net.\n";

	//Training function
	trainFromData(net);
	//Save work
	cout << "Saving net and making image..." << endl;
	cout.flush();
	net.saveto(filedest);
	//Generate image
	FileWriter ex(IMG_WID, IMG_HEI, net, imgdest);
}
void trainFromData(NeuralNet &net) {
	//Sample evaluations
	Matrix out = net.eval(Matrix(1, 2, new double[2] { 0, 0 }));
	cout << "Pre eval at 0, 0:" << endl << out;
	Matrix out2 = net.eval(Matrix(1, 2, new double[2] { IMG_WID, IMG_HEI }));
	cout << "Pre eval at " << IMG_WID << ", " << IMG_HEI << ":" << endl << out2;
	cout << "Loading data...\n";
	cout.flush();

	//Read data from file
	FileReader import("input/training.txt");
	cout << "Loaded training data.\nBeginning training...\n";
	cout.flush();

	//Train network
	net.train(import.getInputs(), import.getOutputs());
	cout << "Training complete.\n";
	cout.flush();
	cout << net << endl;
	cout.flush();

	//Sample evaluation
	Matrix out3 = net.eval(Matrix(1, 2, new double[2] { 0, 0 }));
	cout << "Post eval at 0, 0:\n" << out3;
	Matrix out4 = net.eval(Matrix(1, 2, new double[2] { IMG_WID, IMG_HEI }));
	cout << "Post eval at " << IMG_WID << ", " << IMG_HEI << ":" << endl << out4;
}
