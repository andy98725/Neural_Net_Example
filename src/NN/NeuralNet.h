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
#include <thread> //Multithreading
class NeuralNet
{
private:
  int layers;
  int inCount, outCount, hidCount;
  vector<Matrix> weights;
  vector<Matrix> bases;
  vector<Matrix> values;
  vector<Matrix> activations;
  void resetEval();
public:
  //Declare randomized/ "new" neural net
  NeuralNet (int inCount, int outCount, int hiddenCount, int layers);
  //Load from file
  NeuralNet (string);
  //Destructor
  virtual
  ~NeuralNet ();
  //Evaluate input
  Matrix eval(Matrix);
  //Generic "Train". Set to whatever the developer thinks fits
  void train(vector<Matrix>, vector<Matrix>);
  //Backpropogation is the base learning method
  void backprop(vector<Matrix>, vector<Matrix>, float);
  //Training method of selection sampling
  void escelate(vector<Matrix>, vector<Matrix>, int);

  //Save neural net's current state to file
  void saveto (string);
};

#endif /* NEURALNET_H_ */
