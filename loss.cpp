#include "loss.h"

double MSE(const Vector& prediction, const Vector& target) {
  double result{0.0};
  int n = prediction.size;

  for (int i{0}; i < n; i++){
    result += (prediction(i)-target(i))*(prediction(i)-target(i));
  }

  return 1/n * result;
}

double MAE(const Vector& prediction, const Vector& target) {
  double result{0.0};
  int n = prediction.size;

  for (int i{0}; i < n; i++){
    result += std::abs(prediction(i)-target(i));
  }

  return 1/n * result;
}
