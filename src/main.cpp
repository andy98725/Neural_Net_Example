/*	Improved Neural Net by Andy Hudson
 *
 */
#include <iostream>

#include "math/Matrix.h"
#include "import.h"
using namespace std;

int main(){
//  Matrix a({{2,1},{1,2}});
//  cout << a << endl;
  FileReader import("training.txt");
  const int SIZE = import.getSize();
  Matrix* inputs = import.getInputs();
  Matrix* outputs = import.getOutputs();
  for(int i = 0; i < 10; i++)
  //cout << inputs[i] << endl << outputs[i] << endl;
  return 0;
}
