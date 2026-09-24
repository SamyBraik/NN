#include "activation.h"
#include <cmath>

Vector ReLU(const Vector& x) {
  Vector z(x.size);

  for (int i{0}; i < x.size; i++){
    if (x(i)>0) z(i)=x(i);
    else z(i)=0;
  }

  return z;
}

Vector ReLU_derivative (const Vector& x) {
  Vector z(x.size);
  
  for (int i{0}; i < x.size; i++){
    if (x(i)>0) z(i) = 1;
    else z(i) = 0;
  }
  return z;
}

Matrix ReLU (const Matrix& matrix) {
  Matrix result(matrix.rows, matrix.cols);

  for (int i{0}; i < matrix.rows*matrix.cols; i++){
    if (matrix.data[i]>0) result.data[i] = matrix.data[i];
    else result.data[i] = 0;
  }
  return result;
}

Matrix ReLU_derivative (const Matrix& matrix){
  Matrix result(matrix.rows, matrix.cols);

  for (int i{0}; i < matrix.rows*matrix.cols; i++){
    if (matrix.data[i]>0) result.data[i] = 1;
    else result.data[i] = 0;
  }
  return result;
}

Vector Sigmoide (const Vector& x) {
  Vector z(x.size);

  for (int i{0}; i < x.size; i++){
    z(i) = 1/(1+std::exp(-x(i)));
  }

  return z;
}

Vector Sigmoide_derivative (const Vector& x) {
  Vector s = Sigmoide(x);
  Vector z(x.size);

  for (int i{0}; i < x.size; i++){
    z(i) = s(i) * (1 - s(i));
  } 

  return z;
}

Vector Tanh (const Vector& x) {
  Vector z(x.size);

  for (int i{0}; i < x.size; i++){
    z(i) = std::tanh(x(i));
  }

  return z;
}

Vector Tanh_derivative (const Vector& x) {
  Vector t = Tanh(x);
  Vector z(x.size);

  for (int i{0}; i < x.size; i++){
    z(i) = 1 - t(i)*t(i);
  }

  return z;
}
