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

class Matrix
{
private:
  unsigned int r, c;
  vector<float> arr;
public:
  //Default/Dimension constructor
  Matrix(unsigned int r = 1, unsigned int c = 1, float v = 0);
  //Array initializer
  Matrix(int, int, float*);
  //Copy constructor
  Matrix(const Matrix&);
  //Bracket constructor
  Matrix(initializer_list<initializer_list<float>>);
  //String constructor
  Matrix(string);
  //Destructor
  virtual
  ~Matrix ();
  //Get size
  unsigned int getr();
  unsigned int getc();
  //Get value
  float get(unsigned int, unsigned int);
  //Set value
  void set(unsigned int, unsigned int, float);
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

  //Load from string
  void read(string);
  //Save to string
  string write();
};
//Output
ostream &operator<<(ostream&, Matrix&);

#endif /* MATRIX_H_ */
