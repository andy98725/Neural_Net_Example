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
#include "export.h"
#include "math/Matrix.h"
#include "NN/NeuralNet.h"
using namespace std;

void trainFromData(NeuralNet&);
int main(){
  //Load NN from file
  NeuralNet baby("NNets/beginnerNet.nn");

  cout << "Initiated net.\n";
  //Sample evaluation
  Matrix out = baby.eval(Matrix{{0,0}});
  cout << "Pre eval at 0,0:\n" << out;
  cout << "Training net...\n";
  //Training function
  trainFromData(baby);
  //Sample evaluation
  out = baby.eval(Matrix{{0,0}});
  cout << "Post eval at 0,0:\n" << out;
  //Save work
  baby.saveto("NNets/beginnerNet.nn");
  //Generate image
  //FileWriter ex(469,376, baby, "out.txt");
  return 0;
}
void trainFromData(NeuralNet &net){
  //Read data from file
  FileReader import("training.txt");
  cout << "Loaded training data.\nBeginning training...\n";
  //Train network
  net.train(import.getInputs(),import.getOutputs());
  cout << "Training complete.\n";
}
