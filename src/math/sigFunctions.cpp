/*
 * sigFunctions.cpp
 *
 *  Created on: Mar 2, 2018
 *      Author: Andy
 */

#include "sigFunctions.h"

float sigmoid(float in){
  return 1 / (1 + exp (-in));
}

//float sigmoidPrime(float in){
//  return sigmoid(in) * (1-sigmoid(in));
//} Same as below

float sigmoidPrime(float in){
	return exp(-in) / pow(1+exp(-in),2);
}

