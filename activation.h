#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "algebra.h"

Vector ReLU (const Vector& x);
Matrix ReLU (const Matrix& matrix);
Vector ReLU_derivative(const Vector& x);
Matrix ReLU_derivative(const Matrix& matrix);

Vector sigmoide(const Vector& x);
Vector sigmoide_derivative(const Vector& x);

Vector tanh(const Vector& x);
Vector tanh_derivative(const Vector& x);

#endif 
