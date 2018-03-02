/*
 * matrix.h
 *
 *  Created on: Feb 26, 2018
 *      Author: Andy
 */

#ifndef MATRIX_H_
#define MATRIX_H_
using namespace std;

#include <string>
#include <stdexcept>

class Matrix
{
private:
  int r, c;
  float *arr;
public:
  //Default/Dimension constructor
  Matrix(int r = 1, int c = 1, float v = 0);
  //Array initializer
  Matrix(int, int, float*);
  //Copy constructor
  Matrix(const Matrix&);
  //Bracket constructor
  Matrix(std::initializer_list<std::initializer_list<float>>);
  //Destructor
  virtual
  ~Matrix ();
  //Get value
  float get(int, int);
  //Set value
  void set(int, int, float);
  //Transpose
  Matrix transpose();
  //Readable output
  std::string getString();
  //Operator overloads
  Matrix operator+(const Matrix&) const;
  Matrix operator-(const Matrix&) const;
  Matrix operator*(const Matrix&) const; //Dot product
  Matrix operator%(const Matrix&) const; //Hadamard product
  Matrix operator*(const float) const; //Scalar mult
  Matrix operator/(const float) const; //Scalar div

  //Matrixwise apply function
  Matrix apply(float (*)(float));
};
//Output
ostream &operator<<(ostream&, Matrix&);

#endif /* MATRIX_H_ */
