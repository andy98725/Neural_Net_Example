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

int main(){
  //Read data
  cout << "Initiated.\n";
  FileReader import("training.txt");
  cout << "Loaded training data.\n";
  vector<Matrix> inputs = import.getInputs();
  vector<Matrix> outputs = import.getOutputs();
  //Neural Net training
  NeuralNet network(2,3,8,5);
  Matrix out = network.eval(inputs[100]);
  cout << "Loaded network. Sample evaluation:\nIN: " << inputs[100] << "OUT: " << out;
  cout <<"Training...\n";
  network.train(inputs,outputs);
  cout << "Evaluating...\n";
  out = network.eval(inputs[100]);
  cout << out << endl;
  return 0;
}
