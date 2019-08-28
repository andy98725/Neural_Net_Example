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
#include <math.h> // isnan

class Matrix {
private:
	unsigned int r, c;
	long double* arr;
public:
	//Default/Dimension constructor
	Matrix(unsigned int r, unsigned int c, long double v);
	//Array initializer
	Matrix(int, int, long double*);
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
	long double get(unsigned int, unsigned int);
	//Set value
	void set(unsigned int, unsigned int, long double);
	//Transpose
	void transpose();
	//Readable output
	std::string toString();
	//Operator overloads
	void operator+=(const Matrix&);
	void operator-=(const Matrix&);
	void operator*=(const Matrix&); //Dot product
	Matrix operator*(const Matrix&) const;
	void operator%=(const Matrix&); //Hadamard product

	void operator*=(const double); //Scalar mult
	void operator/=(const double); //Scalar divide

	//Matrixwise apply function
	void apply(long double (*)(long double));

	//Save to string
	string write();
};
//Output
ostream &operator<<(ostream&, Matrix&);

// Contains a bad value?
bool containsNaN(Matrix&);

#endif /* MATRIX_H_ */
