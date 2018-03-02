/*
 * import.cpp
 *
 *  Created on: Feb 28, 2018
 *      Author: Andy
 */


#include "import.h"

using namespace std;

FileReader::FileReader(string filename){
  ifstream file(filename);
  if(!file.is_open()){
      throw runtime_error("File at "+filename+"not found");
  }
  int wid, hei;
  file >> wid >> hei;
  size = 10;//wid*hei;
  inputs = new Matrix[size];
  outputs = new Matrix[size];
  int index = 0;
  for(string line; getline(file, line);){
      istringstream read(line);
      float* iarr = new float[2];
      read >> iarr[0] >> iarr[1];
      float* oarr = new float[3];
      read >> oarr[0] >> oarr[1] >> oarr[2];
      inputs[index] = Matrix(1,2,iarr);
      outputs[index] = Matrix(1,3,oarr);
      cout << oarr[0]<< endl;
      index++;
      if(index == 10) exit(0);

  }
  /* OLD
  for(int count = 0; count < size; ++count){
      int i, j;
      file >> i >> j;
      float *iarr = new float[2];
      iarr[0] = (float)i;
      iarr[1] = (float)j;

      inputs[count] = Matrix(1,2,iarr);
      float r, g, b;
      file >> r >> g >> b;
      float* oarr = new float[3];
      oarr[0] = r;
      oarr[1] = g;
      oarr[2] = b;
      outputs[count] = Matrix(1,3,oarr);
      cout << inputs[count]<< endl << outputs[count] << endl << endl;
  }*/
}

int FileReader::getSize(){
  return size;
}

Matrix* FileReader::getInputs(){
  return inputs;
}
Matrix* FileReader::getOutputs(){
  return outputs;
}
