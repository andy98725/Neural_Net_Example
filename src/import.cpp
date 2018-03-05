/*
 * import.cpp
 *
 *  Created on: Feb 28, 2018
 *      Author: Andy
 */

#include "import.h"

using namespace std;

FileReader::FileReader (string filename)
{
  ifstream file (filename);
  if (!file.is_open ())
    {
      throw runtime_error ("File at " + filename + "not found");
    }
  for (string line; getline (file, line);)
    {
      istringstream read (line); //Main loop
      float* iarr = new float[2];
      float* oarr = new float[3];
      read >> iarr[0] >> iarr[1] >> oarr[0] >> oarr[1] >> oarr[2];
      inputs.push_back(Matrix(1,2,iarr));
      outputs.push_back(Matrix(1,3,oarr));
      delete iarr;
      delete oarr;
    }
  file.close ();
}


vector<Matrix>
FileReader::getInputs ()
{
  return inputs;
}
vector<Matrix>
FileReader::getOutputs ()
{
  return outputs;
}
