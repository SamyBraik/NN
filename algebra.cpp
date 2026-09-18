#include "algebra.h"
#include <algorithm>
#include <iostream>



Vector::Vector(int size) : size(size), data(size, 0.0) {}

double& Vector::operator()(int i) {return data[i];}
double Vector::operator()(int i) const {return data[i];}

//Vector product element-wise
Vector Vector::hadamard (const Vector& v) const {
  Vector result(size);

  for (int i{0}; i < size; i++){
    result(i) = data[i]*v(i);
  }

  return result;
}
// Vector dot product
double Vector::operator* (const Vector& v) const {
  double result{0.0};

  for (int i{0}; i < size; i++){
    result += data[i]*v(i);
  }

  return result;
}

//Vector and scalar product
Vector Vector::operator* (const double& lambda) const{
  Vector result(size);

  for (int i{0}; i < size; i++){
    result(i) = lambda * data[i];
  }

  return result;
}

// Vector addition
Vector Vector::operator+ (const Vector& v) const {
  Vector result(size);

  for (int i{0}; i < size; i++){
    result(i) = data[i] + v(i);
  }

  return result;
}

//Vector substraction
Vector Vector::operator- (const Vector& v) const {
  Vector result(size);

  for (int i{0}; i < size; i++){
    result(i) = data[i] - v(i);
  }

  return result;
}

// Vector substraction with scalar
Vector Vector::operator- (const double& lambda) const{
  Vector result(size);

  for (int i{0}; i < size; i++){
    result(i) = data[i] - lambda;
  }

  return result;
}

//Tensor product
Matrix Vector::tensor(const Vector& v) const {
  Matrix result(size,v.size);

  for (int i{0}; i < size; i++){
    for (int j{0}; j < v.size; j++){
      result(i,j) = data[i]*v(j);
    }
  }
  return result;
}



Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows*cols, 0.0) {}

double& Matrix::operator()(int r, int c){return data[r*cols+c];}
double Matrix::operator()(int r, int c) const {return data[r*cols +c];}

//Matrix transposition
Matrix Matrix::transpose() const{
  Matrix result(cols, rows);
  
  for (int i{0}; i < cols; i++){
    for (int j{0}; j < rows; j++){
      result(i,j) = data[j*cols+i];
    }
  }
  return result;
}

// Matrix multiplication using tiling
Matrix Matrix::operator*(const Matrix& other) const{
  Matrix result(rows, other.cols);
  if (cols != other.rows) {std::cout<<"Dimension mismatch"; return result;} //bof bof cette ligne
  
  int T = 32;
  for (int i0{0}; i0 < rows; i0 += T)
    for (int j0{0}; j0 < other.cols; j0 += T)
      for (int k0{0}; k0 < cols; k0 += T)
        for (int i{0}; i < std::min(i0 + T, rows); i++)
          for (int j{0}; j < std::min(j0 + T, other.cols); j++)
            for (int k{0}; k < std::min(k0 + T, cols); k++)
              result(i,j) += (*this)(i,k) * other(k,j);

  return result;
}

// Matrix addition
Matrix Matrix::operator+(const Matrix& other) const{
  Matrix result(rows, cols);

  for (int i{0}; i < rows; i++)
    for (int j{0}; j <cols; j++)
      result(i,j) = (*this)(i,j)+other(i,j);

  return result;
}

// Vector and matrix product
Vector Matrix::operator*(const Vector& v) const {
  Vector result(rows);

  for (int i{0}; i<rows;i++)
    for (int j{0}; j<cols;j++)
      result(i) += (*this)(i,j)*v(j);

  return result;
}

//Matrix and scalar product
Matrix Matrix::operator*(const double& lambda) const{
  Matrix result(rows,cols);

  for (int i{0}; i < rows; i++){
    for (int j{0}; j < cols; j++){
      result(i,j) = (*this)(i,j) * lambda;
    }
  }

  return result;
}
 


