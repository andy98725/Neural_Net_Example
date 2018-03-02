/*
 * import.h
 *
 *  Created on: Feb 28, 2018
 *      Author: Andy
 */

#ifndef IMPORT_H_
#define IMPORT_H_

#include <string>
#include <sstream>
#include<fstream>
#include <stdexcept>
#include <iostream>

#include "math/Matrix.h"
using namespace std;

class FileReader{
private:
  int size;
  Matrix* inputs;
  Matrix* outputs;
public:
  FileReader(string);
  int getSize();
  Matrix* getInputs();
  Matrix* getOutputs();

};



#endif /* IMPORT_H_ */
