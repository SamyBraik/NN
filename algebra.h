#ifndef ALGEBRA_H
#define ALGEBRA_H

#include <vector>
#include <cmath>

class Matrix;


class Vector {
  public : 
    Vector(int size);

    double& operator()(int i);
    double operator()(int i) const;

    Vector hadamard(const Vector& v) const;
    double operator*(const Vector& v) const;
    Vector operator*(double lambda) const;
    Vector operator+(const Vector& v) const;
    Vector operator+(double lambda) const;
    Vector operator-(const Vector& v) const;
    Vector operator-(double lambda) const;
    Matrix tensor(const Vector& v) const;
    Vector operator/(double lambda) const;
    Vector operator/(const Vector& v) const;
    Vector power(double lambda) const;
    Matrix unflatten(int rows, int cols) const;

    int size;
    std::vector<double> data;
};

class Matrix {
  public : 
    Matrix(int rows,int cols);
    
    double& operator()(int r, int c);
    double operator()(int r, int c) const;

    Matrix transpose() const;
    Matrix operator*(const Matrix& other) const;
    Matrix hadamard(const Matrix& other) const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator+(double lambda) const;
    Matrix operator-(const Matrix& other) const;
    Vector operator*(const Vector& v) const;
    Matrix operator*(double lambda) const;
    Matrix power(double lambda) const;
    Matrix operator/(double lambda) const;
    Matrix operator/(const Matrix& other) const;

    Matrix max_pooling(int k, int stride, int padding) const;
    Matrix mean_pooling(int k, int stride, int padding) const;
    Matrix convolution(const Matrix& kernel, int stride, int padding) const;
    Vector flatten() const;
 
    int rows, cols;
    std::vector<double> data;
};

#endif
