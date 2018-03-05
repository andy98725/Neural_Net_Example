/*
 * matrix.cpp
 *
 *  Created on: Feb 26, 2018
 *      Author: Andy
 */


#include "Matrix.h"


using namespace std;
//Default/dimension constructor
Matrix::Matrix (int r, int c, float v)
{
  this->r = r;
  this->c = c;
  for(int i = 0; i < r*c; i++){
      arr.push_back(v);
  }


}
//Array constructor
Matrix::Matrix(int r, int c, float* temparr){
  this->r = r;
  this->c = c;
  for(int i = 0; i < r*c; ++i){
      arr.push_back(temparr[i]);
  }
}
//Copy constructor
Matrix::Matrix(const Matrix& m){
  this->r = m.r;
  this->c = m.c;
  for(int i = 0; i < r*c; i++){
      arr.push_back(m.arr[i]);
  }

}
//Bracket constructor
Matrix::Matrix(initializer_list<initializer_list<float>> l){
  r = l.size();
  c = -1; //Not defined yet
  int i = 0; //index
  for(initializer_list<float> ll : l){
      if(c == -1){
	  //Finish definition
	  c = ll.size();
      }
      for(float v : ll){
      arr.push_back(v);
      }
  }
  //Catch poorly structured lists
  if(i != r*c){
      throw invalid_argument("matrix poorly constructed: mismatched initialization lists.");
  }
}
//Destructor
Matrix::~Matrix ()
{
  arr.clear();
}
//get
float Matrix::get(int row, int col){
  //Catch out of bounds
  if(row < 0 || col < 0 || row >= r || col >= c){
      string vals = "Index "+to_string(row)+','+to_string(col)+" called on matrix size of "+to_string(r)+','+to_string(c)+'.';
      throw invalid_argument( "matrix get function called out of range: " + vals );
  }
  return arr[row*c+col];
}
//set
void Matrix::set(int row, int col, float v){
  //Catch out of bounds
  if(row < 0 || col < 0 || row >= r || col >= c){
      string vals = "Index "+to_string(row)+','+to_string(col)+" called on matrix size of "+to_string(r)+','+to_string(c)+'.';
      throw invalid_argument( "matrix set function called out of range: " + vals );
  }
  arr[row*c+col] = v;
}
//transpose
Matrix Matrix::transpose(){
  float* tr = new float[r*c];
  for(int i = 0; i < r; i++){
      for(int j = 0; j < c; j++){
	  tr[i*c+j] = arr[j*r + i];
      }
  }
  return Matrix(c, r, tr);
}
//String output
string Matrix::getString(){
  string ret = "{";
  for(int i = 0; i < r; i++){
      ret += '{';
      for(int j = 0; j < c; j++){
	  ret += to_string(arr[i*c+j]);
	  if(j+1 < c) ret += ", ";
      }
      ret += '}';
      if(i+1 < r) ret += ",\n";
  }
  ret += "}\n";
  return ret;
}
//String output
ostream &operator<<(ostream& os, Matrix& r){
  os << r.getString();
  return os;
}

//Operator overload
Matrix Matrix::operator + (const Matrix& m) const{
  if(r != m.r || c != m.c){
      string vals = "matrix of dimensions "+to_string(r)+','+to_string(c)+" mismatched with matrix of dimensions "+to_string(m.r)+','+to_string(m.c)+'.';
      throw invalid_argument("mismatched matricies add failure: " + vals);
  }
  float* add = new float[r*c];
  for(int i = 0; i < r*c; i++){
      add[i] = arr[i] + m.arr[i];
  }
  return Matrix(r, c, add);

}
Matrix Matrix::operator - (const Matrix& m) const{
  if(r != m.r || c != m.c){
      string vals = "matrix of dimensions "+to_string(r)+','+to_string(c)+" mismatched with matrix of dimensions "+to_string(m.r)+','+to_string(m.c)+'.';
      throw invalid_argument("mismatched matricies subtract failure: " + vals);
  }
  float* sub = new float[r*c];
  for(int i = 0; i < r*c; i++){
      sub[i] = arr[i] - m.arr[i];
  }
  return Matrix(r, c, sub);

}
Matrix Matrix::operator * (float m) const{
  float* mult = new float[r*c];
  for(int i = 0; i < r*c; i++){
      mult[i] = arr[i] * m;
  }
  return Matrix(r, c, mult);

}
Matrix Matrix::operator / (float d) const{
  if(d == 0) throw invalid_argument("Operation error: cannot divide matrix by 0");
  float* div = new float[r*c];
  for(int i = 0; i < r*c; i++){
      div[i] = arr[i] / d;
  }
  return Matrix(r, c, div);

}
//Dot product
Matrix Matrix::operator * (const Matrix& m) const{
  if(c != m.r){
      string vals = "matrix of dimensions "+to_string(r)+','+to_string(c)+" mismatched with matrix of dimensions "+to_string(m.r)+','+to_string(m.c)+'.';
      throw invalid_argument("mismatched matricies dot product failure: " + vals);
  }
  float *prod = new float[r*m.c];
  for(int i = 0; i < r; i++){
      for(int j = 0; j < m.c; j++){
	  prod[i*m.c+j] = 0;
	  for(int k = 0; k < c; k++){
	      prod[i*m.c+j] += arr[i*c+k] * m.arr[k*m.c+j];
	  }
      }
  }
  return Matrix(r, m.c, prod);
}
//Hadamard product
Matrix Matrix::operator % (const Matrix& m) const{
  if(r != m.r || c != m.c){
      string vals = "matrix of dimensions "+to_string(r)+','+to_string(c)+" mismatched with matrix of dimensions "+to_string(m.r)+','+to_string(m.c)+'.';
      throw invalid_argument("mismatched matricies hadamard failure: " + vals);
  }
  float* had = new float[r*c];
  for(int i = 0; i < r*c; i++){
      had[i] = arr[i] * m.arr[i];
  }
  return Matrix(r, c, had);

}
//Function application
Matrix Matrix::apply(float (*func)(float)){
  float* out = new float[r*c];
  for(int i = 0; i < r*c; i++){
      out[i] = func(arr[i]);
  }
  return Matrix(r, c, out);
}
