/*
 * matrix.h
 *
 *  Created on: Feb 26, 2018
 *      Author: Andy
 */

#ifndef MATRIX_H_
#define MATRIX_H_
using namespace std;

#include <string> //string (constructors, etc)
#include <stdexcept> //throws (constructors, etc)
#include <iostream> //cout (bugtesting)
#include <sstream> //stringstream (read, write)
#include <vector> //vector arr (data)

class Matrix {
private:
	unsigned int r, c;
	float* arr;
public:
	//Default/Dimension constructor
	Matrix(unsigned int r, unsigned int c, float v);
	//Array initializer
	Matrix(int, int, float*);
	//Copy constructor
	Matrix(const Matrix&);
	//String constructor
	Matrix(string);
	//Destructor
	virtual
	~Matrix();
	//Get size
	unsigned int getr();
	unsigned int getc();
	//Get value
	float get(unsigned int, unsigned int);
	//Set value
	void set(unsigned int, unsigned int, float);
	//Transpose
	void transpose();
	//Readable output
	std::string getString();
	//Operator overloads
	void operator+=(const Matrix&);
	void operator-=(const Matrix&);
	void operator*=(const Matrix&); //Dot product
	void operator%=(const Matrix&); //Hadamard product

	void operator*=(const float); //Scalar mult
	void operator/=(const float); //Scalar divide

	//Matrixwise apply function
	void apply(float (*)(float));

	//Save to string
	string write();
};
//Output
ostream &operator<<(ostream&, Matrix&);

#endif /* MATRIX_H_ */
