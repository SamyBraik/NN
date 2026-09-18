#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "algebra.h"

Vector ReLU (const Vector& x);
Vector ReLU_derivative(const Vector& x);

Vector sigmoide(const Vector& x);
Vector sigmoide_derivative(const Vector& x);

Vector tanh(const Vector& x);
Vector tanh_derivative(const Vector& x);

#endif 
