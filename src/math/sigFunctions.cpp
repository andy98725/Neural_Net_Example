/*
 * sigFunctions.cpp
 *
 *  Created on: Mar 2, 2018
 *      Author: Andy
 */

#include "sigFunctions.h"

long double sigmoid(long double in) {
	return 1 / (1 + exp(-in));
}

long double sigmoidPrime(long double in) {
	return exp(-in) / pow(1 + exp(-in), 2);
}

