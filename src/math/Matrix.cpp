/*
 * matrix.cpp
 *
 *  Created on: Feb 26, 2018
 *      Author: Andy
 */

using namespace std;

#include "Matrix.h"

//Default/dimension constructor
Matrix::Matrix(unsigned int r, unsigned int c, float v) {
	this->r = r;
	this->c = c;
	this->arr = new float[r * c];
	for (unsigned int i = 0; i < r * c; i++) {
		arr[i] = v;
	}

}
//Array constructor
Matrix::Matrix(int r, int c, float* arr) {
	this->r = r;
	this->c = c;
	this->arr = arr;
}
//Copy constructor
Matrix::Matrix(const Matrix& m) {
	this->r = m.r;
	this->c = m.c;
	this->arr = new float[r * c];

	for (unsigned int i = 0; i < r * c; i++) {
		arr[i] = m.arr[i];
	}

}
//Destructor
Matrix::~Matrix() {
	delete arr;
}
//get dimensions
unsigned int Matrix::getr() {
	return r;
}
unsigned int Matrix::getc() {
	return c;
}
//get values
float Matrix::get(unsigned int row, unsigned int col) {
	//Catch out of bounds
	if (row < 0 || col < 0 || row >= r || col >= c) {
		string vals = "Index " + to_string(row) + ',' + to_string(col)
				+ " called on matrix size of " + to_string(r) + ','
				+ to_string(c) + '.';
		throw invalid_argument(
				"matrix get function called out of range: " + vals);
	}
	return arr[row * c + col];
}
//set
void Matrix::set(unsigned int row, unsigned int col, float v) {
	//Catch out of bounds
	if (row < 0 || col < 0 || row >= r || col >= c) {
		string vals = "Index " + to_string(row) + ',' + to_string(col)
				+ " called on matrix size of " + to_string(r) + ','
				+ to_string(c) + '.';
		throw invalid_argument(
				"matrix set function called out of range: " + vals);
	}
	arr[row * c + col] = v;
}
//transpose
void Matrix::transpose() {
	float* tr = new float[r * c];
	for (unsigned int i = 0; i < r; i++) {
		for (unsigned int j = 0; j < c; j++) {
			tr[i * c + j] = arr[j * r + i];
		}
	}
	// Update existing
	delete arr;
	arr = tr;
	int temp = r;
	r = c;
	c = temp;
}
//String output
string Matrix::toString() {
	string ret = "{";
	for (unsigned int i = 0; i < r; i++) {
		ret += '{';
		for (unsigned int j = 0; j < c; j++) {
			ret += to_string(arr[i * c + j]);
			if (j + 1 < c)
				ret += ", ";
		}
		ret += '}';
		if (i + 1 < r)
			ret += ",\n";
	}
	ret += "}\n";
	return ret;
}
//String output
ostream &operator<<(ostream& os, Matrix& r) {
	os << r.toString();
	return os;
}

// Matrix operators
void Matrix::operator+=(const Matrix& m) {
	if (r != m.r || c != m.c) {
		string vals = "matrix of dimensions " + to_string(r) + ','
				+ to_string(c) + " mismatched with matrix of dimensions "
				+ to_string(m.r) + ',' + to_string(m.c) + '.';
		throw invalid_argument("mismatched matricies add failure: " + vals);
	}
	// Add other matrix values
	for (unsigned int i = 0; i < r * c; i++) {
		arr[i] += m.arr[i];
	}
}
void Matrix::operator-=(const Matrix& m) {
	if (r != m.r || c != m.c) {
		string vals = "matrix of dimensions " + to_string(r) + ','
				+ to_string(c) + " mismatched with matrix of dimensions "
				+ to_string(m.r) + ',' + to_string(m.c) + '.';
		throw invalid_argument(
				"mismatched matricies subtract failure: " + vals);
	}
	// Subtract other matrix values
	for (unsigned int i = 0; i < r * c; i++) {
		arr[i] -= m.arr[i];
	}
}
void Matrix::operator*=(float m) {
	for (unsigned int i = 0; i < r * c; i++) {
		arr[i] *= m;
	}
}
void Matrix::operator/=(float d) {
	for (unsigned int i = 0; i < r * c; i++) {
		arr[i] /= d;
	}
}
// Dot product
Matrix Matrix::operator*(const Matrix& m) const{
	if (c != m.r) {
		string vals = "matrix of dimensions " + to_string(r) + ','
				+ to_string(c) + " mismatched with matrix of dimensions "
				+ to_string(m.r) + ',' + to_string(m.c) + '.';
		throw invalid_argument(
				"mismatched matricies dot product failure: " + vals);
	}
	float *prod = new float[r * m.c];
	for (unsigned int i = 0; i < r; i++) {
		for (unsigned int j = 0; j < m.c; j++) {
			prod[i * m.c + j] = 0;
			for (unsigned int k = 0; k < c; k++) {
				prod[i * m.c + j] += arr[i * c + k] * m.arr[k * m.c + j];
			}
		}
	}
	// Make new matrix
	return Matrix(r, m.c, prod);
}
//Dot product locally
void Matrix::operator*=(const Matrix& m) {
	if (c != m.r) {
		string vals = "matrix of dimensions " + to_string(r) + ','
				+ to_string(c) + " mismatched with matrix of dimensions "
				+ to_string(m.r) + ',' + to_string(m.c) + '.';
		throw invalid_argument(
				"mismatched matricies dot product failure: " + vals);
	}
	float *prod = new float[r * m.c];
	for (unsigned int i = 0; i < r; i++) {
		for (unsigned int j = 0; j < m.c; j++) {
			prod[i * m.c + j] = 0;
			for (unsigned int k = 0; k < c; k++) {
				prod[i * m.c + j] += arr[i * c + k] * m.arr[k * m.c + j];
			}
		}
	}
	// Update
	delete arr;
	arr = prod;
	c = m.c;
}
//Hadamard product
void Matrix::operator %=(const Matrix& m) {
	if (r != m.r || c != m.c) {
		string vals = "matrix of dimensions " + to_string(r) + ','
				+ to_string(c) + " mismatched with matrix of dimensions "
				+ to_string(m.r) + ',' + to_string(m.c) + '.';
		throw invalid_argument(
				"mismatched matricies hadamard failure: " + vals);
	}
	for (unsigned int i = 0; i < r * c; i++) {
		arr[i] *= m.arr[i];
	}
}
//Function application
void Matrix::apply(float (*func)(float)) {
	for (unsigned int i = 0; i < r * c; i++) {
		arr[i] = func(arr[i]);
	}
}
// String constructor
Matrix::Matrix(string in) {
	stringstream read(in);
	r = 0;
	c = 0;
	read >> r;
	read >> c;
	arr = new float[r * c];
	for (unsigned int i = 0; i < r * c; ++i) {
		float in;
		read >> in;
		arr[i] = in;
	}
}
// Write to string
string Matrix::write() {
	stringstream ret;
	ret << r << ' ' << c;
	for (unsigned int i = 0; i < r * c; ++i) {
		ret << ' ' << arr[i];
	}
	return ret.str();
}
