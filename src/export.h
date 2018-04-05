/*
 * export.h
 *
 *  Created on: Mar 8, 2018
 *      Author: Andy
 */

#ifndef EXPORT_H_
#define EXPORT_H_

#include "NN/NeuralNet.h"
#include "math/Matrix.h"
#include<fstream>
using namespace std;
class FileWriter{
private:
  int wid;
  int hei;
  NeuralNet net;
public:
  FileWriter(int, int, NeuralNet, string);
};



#endif /* EXPORT_H_ */
