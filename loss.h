#ifndef LOSS_H
#define LOSS_H

#include "algebra.h"
#include <cmath>

double MSE(const Vector& prediction, const Vector& target) ;
double MAE(const Vector& prediction, const Vector& target) ;

#endif
