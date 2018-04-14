/*
 * NeuralNet.cpp
 *
 *  Created on: Feb 28, 2018
 *      Author: Andy
 */

#include "NeuralNet.h"
using namespace std;

NeuralNet::NeuralNet (int inCount, int outCount, int hiddenCount, int layers)
{
  this->layers = layers;
  this->inCount = inCount;
  this->outCount = outCount;
  this->hidCount = hiddenCount;
  //weights = new Matrix[layers];
  for (int i = 0; i < layers; ++i)
    {
      int rows = hidCount, cols = hidCount;
      if (i == 0)
	rows = inCount;
      if (i == layers - 1)
	cols = outCount;
      float* weightarr = new float[rows * cols], *basearr = new float[cols];
      for (int j = 0; j < rows * cols; ++j)
	{
	  weightarr[j] = (double) rand () / (RAND_MAX); //0 to 1
//	  weightarr[j] = 1 - ((double) rand () / (RAND_MAX)) * 2; //-1 to 1
	}
      for (int j = 0; j < cols; ++j)
	{
	  basearr[j] = 1 - ((double) rand () / (RAND_MAX)) * 2; //-1 to 1
	}
      Matrix w (rows, cols, weightarr), b (1, cols, basearr);
      weights.push_back (w);
      bases.push_back (b);
      delete weightarr;
      delete basearr;
    }

}
//Other declarator is by saveto for paralleling

NeuralNet::~NeuralNet ()
{
  for (vector<Matrix> data :
    { weights, bases, values, activations })
    {
      data.clear ();
    }
}

Matrix
NeuralNet::eval (Matrix in)
{
  //Clear for use
  resetEval ();
  //Copy input to values and activations. Activations for input is just the inputs, no sigmoid
  values.push_back (Matrix (in));
  activations.push_back (Matrix (in));
  for (int i = 0; i < layers; ++i)
    {
      values.push_back (activations.back () * weights[i] + bases[i]);
      activations.push_back (values.back ().apply (sigmoid));
    }
  //Copy output activations
  return Matrix (activations.back ());
}
void
NeuralNet::resetEval ()
{
  for (vector<Matrix> data :
    { values, activations })
    {
      data.clear ();
    }
}
void
NeuralNet::resetErrors ()
{
  weightsError.clear ();
  basesError.clear ();
  for (int i = 0; i < layers; ++i)
    {
      int r, c;
      r = weights[i].getr ();
      c = weights[i].getc ();
      float* weightarr = new float[r * c], *basearr = new float[c];
      for (int j = 0; j < r * c; ++j)
	{
	  weightarr[j] = 0;
	}
      for (int j = 0; j < c; ++j)
	{
	  basearr[j] = 0;
	}
      weightsError.push_back (Matrix (r, c, weightarr));
      basesError.push_back (Matrix (1, c, basearr));
    }
}
void
NeuralNet::backpropCase (Matrix in, Matrix expectedOut)
{
  //Make case errrors matching weight and base error matricies
  vector<Matrix> caseWeightError, caseBaseError;
  for (int i = 0; i < layers; ++i)
    {
      int rows = hidCount, cols = hidCount;
      if (i == 0)
	rows = inCount;
      if (i == layers - 1)
	cols = outCount;
      float* weightarr = new float[rows * cols], *basearr = new float[cols];
      for (int j = 0; j < rows * cols; ++j)
	{
	  weightarr[j] = 0;
	}
      for (int j = 0; j < cols; ++j)
	{
	  basearr[j] = 0;
	}
      caseWeightError.push_back(Matrix(rows,cols,weightarr));
      caseBaseError.push_back(Matrix(1,cols,basearr));
    }
  //Now, begin actual backpropogation. Get the output
  Matrix output = eval(in);
  //Backprop
        for (int layer = layers - 1; layer >= 0; --layer)
  	{
  	  Matrix layerError;
  	  if (layer == layers - 1) //For final layer, error is the difference of output/expected output times the rate of change
  	    layerError = (output - expectedOut);
  	  else
  	    //For other layers, The error is the error of the next layer dot the weights of that layer
  	    layerError = Matrix (
  		caseBaseError[layer + 1] * (weights[layer + 1].transpose ()));
  	  //Finally, we have to hadamard the layer with the rate of change of that node
  	  layerError = layerError % (values[layer + 1].apply (sigmoidPrime));
  	  //Now we have the layer's error.
  	  //The error of the base is actually equal to the layer's error
  	  caseBaseError[layer] = Matrix (layerError);
  	  //And the weight error is the previous layer's activations transposed dot the layer's error
  	  caseWeightError[layer] = Matrix (
  	      activations[layer].transpose () * layerError);
  	}
        //Done backpropogating. Apply case to total and finish.
        for (int j = 0; j < layers; ++j)
  	{
  	  basesError[j] = basesError[j] + caseBaseError[j];
  	weightsError[j] = weightsError[j] + caseWeightError[j];
  	}
}
void
NeuralNet::backprop (vector<Matrix> ins, vector<Matrix> outs, float delta)
{
  //Evaluate and backpropogate through each and every data set.
  //Once done, alter weights and biases according to average error.

  //Catch bad cases
  if (ins.size () == 0)
    throw invalid_argument ("Backprop failure: Empty input vector.");
  if (ins.size () != outs.size ())
    throw invalid_argument (
	"Backprop failure: Mismatched input and output vectors.");
  //clear error matricies
  resetErrors ();


  //TODO: Finish setting up backpropogation multithreading.
  //Init threads
  //Loop through each data point and thread the bacpropogation (TODO)
  for (unsigned int i = 0; i < ins.size (); ++i)
    {
      backpropCase(ins[i],outs[i]);
    }
  //Done evaluating. Clear then apply average errors.
  resetEval ();
  for (unsigned int i = 0; i < weights.size (); ++i)
    {
      weights[i] = weights[i] + (weightsError[i] / ins.size ()) * delta;
    }
  for (unsigned int i = 0; i < bases.size (); ++i)
    {
      bases[i] = bases[i] + (basesError[i] / ins.size ()) * delta;
    }
}
void
NeuralNet::escelate (vector<Matrix> ins, vector<Matrix> outs, int times)
{
  //Trains N times
  const int MAX = times;
  for (int t = 0; t < MAX; ++t)
    {
      vector<Matrix> inset, outset;
      //Get 10 random test cases for each iteration that's happened, plus base 20 (escalating count)
      for (int iter = 0; iter < 20 + t * 10; ++iter)
	{
	  //Choose a random from base set and add it
	  int index = rand () % ins.size ();
	  inset.push_back (ins[index]);
	  outset.push_back (outs[index]);
	}
      //Now, backprop set with offset
      float offset = 0.25 - (0.2 * t) / MAX; //Deescelating for accuracy
      backprop (inset, outset, offset);
    }
}
void
NeuralNet::train (vector<Matrix> ins, vector<Matrix> outs)
{
  for (int k = 0; k < 10; ++k){
      escelate (ins, outs, 10);
  }
}
//Load from file
NeuralNet::NeuralNet (string filename)
{
  ifstream file (filename);
  for (string line; getline (file, line);)
    {
      Matrix a (line);
      weights.push_back (a);
      getline (file, line);
      Matrix b (line);
      bases.push_back (b);
    }
  layers = weights.size ();
  inCount = weights[0].getr ();
  hidCount = weights[0].getc ();
  outCount = weights[layers - 1].getc ();
  file.close ();

}
//Save to file
void
NeuralNet::saveto (string filename)
{
  ofstream file (filename);
  for (unsigned int i = 0; i < weights.size (); ++i)
    {
      file << (weights[i].write ()) << '\n' << (bases[i].write ());
      if (i + 1 < weights.size ())
	file << endl;
    }
  file.close ();

}
