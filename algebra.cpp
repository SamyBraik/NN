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
Vector Vector::operator* (double lambda) const{
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

Vector Vector::operator+(double lambda) const {
  Vector result(size);

  for (int i{0}; i < size; i++){
    result(i) = data[i] + lambda;
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
Vector Vector::operator- (double lambda) const{
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

Vector Vector::operator/(double lambda) const {
  Vector result(size);

  for (int i{0}; i < size; i++){
    result(i) = data[i]/lambda;
  }

  return result;
}

Vector Vector::operator/(const Vector& v) const {
  Vector result(size);

  for (int i{0}; i < size; i++){
    result(i) = data[i]/v(i);
  }

  return result;
}

Vector Vector::power(double lambda) const {
  Vector result(size);

  for (int i{0}; i < size; i++){
    result(i) = std::pow(data[i], lambda);
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

Matrix Matrix::hadamard(const Matrix& other) const {
  Matrix result(rows, cols);

    for(int i{0}; i < rows; i++){
      for(int j{0}; j < cols; j++){
        result(i,j) = (*this)(i,j)*other(i,j);
      }
    }
  return result;
  }


// Matrix addition
Matrix Matrix::operator+(const Matrix& other) const{
  Matrix result(rows, cols);

  for (int i{0}; i < rows; i++)
    for (int j{0}; j < cols; j++)
      result(i,j) = (*this)(i,j)+other(i,j);

  return result;
}

Matrix Matrix::operator+(double lambda) const {
  Matrix result(rows,cols);

  for(int i{0}; i < rows; i++)
    for (int j{0}; j < cols; j++)
      result(i,j) = (*this)(i,j)+lambda;

  return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
  Matrix result(rows,cols);

  for (int i{0}; i < rows; i++)
    for (int j{0}; j < cols; j++)
      result(i,j) = (*this)(i,j)-other(i,j);

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
Matrix Matrix::operator*(double lambda) const{
  Matrix result(rows,cols);

  for (int i{0}; i < rows; i++){
    for (int j{0}; j < cols; j++){
      result(i,j) = (*this)(i,j) * lambda;
    }
  }

  return result;
}

Matrix Matrix::power(double lambda) const {
  Matrix result(rows, cols);

  for (int i{0}; i< rows;i++){
    for (int j{0}; j < cols; j++)
      result(i,j) = std::pow((*this)(i,j),lambda);
  }

  return result;
}

Matrix Matrix::operator/(double lambda) const{
  Matrix result(rows, cols);

  for(int i{0}; i < rows; i++){
    for(int j{0}; j < cols; j++){
      result(i,j) = (*this)(i,j)/lambda;
    }
  }

  return result;
}

Matrix Matrix::operator/(const Matrix& other) const {
  Matrix result(rows, cols);

  for (int i{0}; i < rows; i++)
    for (int j{0}; j < cols; j++)
      result(i,j) = (*this)(i,j) / other(i,j);

  return result;
}

Matrix Matrix::max_pooling (int k) const {
  int rows_block = rows / k;
  int cols_block = cols / k;
  Matrix result(rows_block, cols_block);

  int current_rows = 0;
  int current_cols = 0;
   //Tackle only matrixes with rows and cols divisible by k
   
  while (current_rows != rows_block) {
    current_cols = 0;

    while (current_cols != cols_block) {
      int left_row_bound = current_rows * k;
      int left_col_bound = current_cols * k;
      int pool = (*this)(left_row_bound, left_col_bound);

      for (int i{left_row_bound}; i < left_row_bound + k; i++)
        for (int j{left_col_bound}; j < left_col_bound + k; j++)
          if ((*this)(i,j)>pool) pool = (*this)(i,j);

      result(current_rows, current_cols) = pool;
      current_cols++;
    }
    current_rows++;
  }

  return result;
}

Matrix Matrix::mean_pooling(int k) const {
  int rows_block = rows / k;
  int cols_block = cols / k;
  Matrix result(k,k);

  int current_rows = 0;
  int current_cols = 0;

  while (current_rows != rows_block) {
    current_cols = 0;

    while (current_cols != cols_block){
      int left_row_bound = current_rows * k;
      int left_col_bound = current_cols * k;
      int sum = 0;

      for (int i{left_row_bound}; i < left_row_bound + k; i++)
        for (int j{left_col_bound}; j < left_col_bound + k; j++)
          sum += (*this)(i,j);

      result(current_rows, current_cols) = sum / static_cast<double>(k*k);
      current_cols ++;
    }
    current_rows++;
  }
  return result;
}

Matrix Matrix::convolution(const Matrix& kernel, int stride) const {
  int out_rows = (rows - kernel.rows) / stride + 1;
  int out_cols = (cols - kernel.cols) / stride + 1;
  Matrix result(out_rows, out_cols);

  for (int i{0}; i < out_rows; i++){
    for (int j{0}; j < out_cols; j++){
      double sum = 0.0;
      for (int p{0}; p < kernel.rows; p++)
        for (int q{0}; q < kernel.cols; q++)
          sum += (*this)(i*stride + p, j*stride + q) * kernel(p,q);
      result(i,j) = sum;
    }
  }
  return result;
}

