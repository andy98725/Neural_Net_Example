/*
 * export.cpp
 *
 *  Created on: Mar 8, 2018
 *      Author: Andy
 */

#include "export.h"

#include <string>

using namespace std;

FileWriter::FileWriter(int w, int h, NeuralNet nn, string filename) :
		wid(w), hei(h), net(nn) {
	const bool writeSize = true;
	//Write to file
	ofstream file(filename);

	//Write image size
	if (writeSize) {
		file << w << ' ' << h << endl;
	}
	//Generate data
	for (int j = 0; j < h; ++j) {
		for (int i = 0; i < w; ++i) {
			Matrix out = nn.eval(Matrix(1, 2, new long double[2] { i,
					j }));
			file << i << ' ' << j << ' ' << out.get(0, 0) << ' '
					<< out.get(0, 1) << ' ' << out.get(0, 2) << endl;
		}
	}
}
