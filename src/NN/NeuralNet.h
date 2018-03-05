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
  NeuralNet (int inCount, int outCount, int hiddenCount, int layers);
  virtual
  ~NeuralNet ();
  Matrix eval(Matrix);
  void train(vector<Matrix>, vector<Matrix>);
  void backprop(vector<Matrix>, vector<Matrix>, float);

  void escelate(vector<Matrix>, vector<Matrix>, int);
};

#endif /* NEURALNET_H_ */
