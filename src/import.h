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
#include <vector>

#include "math/Matrix.h"
using namespace std;

class FileReader{
private:
  vector<Matrix> inputs;
  vector<Matrix> outputs;
public:
  FileReader(string);
  vector<Matrix> getInputs();
  vector<Matrix> getOutputs();

};



#endif /* IMPORT_H_ */
