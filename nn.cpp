#include "nn.h"
#include <random>
#include <stdexcept>
#include "optimizer.h"
#include <iostream>
#include <numeric>

Layer::Layer(int n_in, int n_out, bool output_layer) : n_in(n_in), n_out(n_out), output_layer(output_layer), weights(n_out, n_in), bias(n_out), grad_weights(n_out, n_in), grad_bias(n_out), last_input(n_in), last_z(n_out) {randomized_weights();}

Vector Layer::forward (const Vector& input){
  Vector z = weights * input + bias;
  last_input = input;
  last_z = z;

  if (output_layer == true) return z;

  return ReLU(z);
}

Vector Layer::backward(const Vector& grad_output) {
  Vector dz(grad_output.size);

  if (output_layer == true) dz = grad_output;
  else dz = grad_output.hadamard(ReLU_derivative(last_z));

  Matrix dW = dz.tensor(last_input);
  Vector db = dz;
  Vector dx = weights.transpose() * dz;

  grad_weights = dW;
  grad_bias = db;

  return dx;
}

void Layer::update_weights(Optimizer& optimizer){
    optimizer.update(weights, grad_weights, bias, grad_bias);
}


void Layer::randomized_weights (){
  static std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<double> dist(-0.5,0.5);
  for (int i{0}; i < n_out; i++){
    for (int j{0}; j < n_in; j++){
      weights(i,j) = dist(gen);
    }
  }
}

MLP::MLP(std::vector<int> layer_size, OptimizerType opt_type, double learning_rate){
  for (size_t i{0}; i < layer_size.size()-1; i++){
    bool is_output_layer = (i == layer_size.size()-2);
    layers.push_back(Layer(layer_size[i],layer_size[i+1], is_output_layer));
  }

  for (auto& layer : layers){
    Optimizer* opt = make_optimizer(opt_type, layer.n_out, layer.n_in, learning_rate);
    optimizers.push_back(opt);
  }
}

Vector MLP::forward(const Vector& input){
  Vector current = input;

  for (size_t i{0}; i < layers.size(); i++){
    current = layers[i].forward(current);
  }

  return current;
}

double MLP::compute_loss(const Vector& prediction, const Vector& target, std::string loss) const{
  if (loss == "MSE") return MSE(prediction, target);
  else if (loss == "MAE") return MAE(prediction, target);
  else return 1e100;
}

void MLP::backward(const Vector& prediction, const Vector& target){
  Vector grad = prediction-target;

  for(int i{static_cast<int>(layers.size())-1}; i >= 0; i--){
    grad = layers[i].backward(grad);
  }
}

void MLP::update_weights(){
  for (size_t i{0}; i < layers.size(); i++){
    layers[i].update_weights(*optimizers[i]);
  }
}

void ConvLayer::randomized_kernel (){
  static std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<double> dist(-0.5,0.5);
  for (int i{0}; i < kernel_size; i++){
    for (int j{0}; j < kernel_size; j++){
      kernel(i,j) = dist(gen);
    }
  }
}


Matrix ConvLayer::forward(const Matrix& input) {
  Matrix z = input.convolution(kernel, stride, padding) + bias;
  last_input = input;
  last_z = z;

  return ReLU(z);
}

//Based on Raphael Cousins's CNN Essentials
Matrix ConvLayer::backward(const Matrix& grad_output) {
  Matrix dz = grad_output.hadamard(ReLU_derivative(last_z));

  Matrix dK(kernel_size, kernel_size);
  for (int k{0}; k < kernel_size; k++){
    for (int l{0}; l < kernel_size; l++){
      double sum = 0.0;
      for (int i{0}; i < dz.rows; i++){
        for (int j{0}; j < dz.cols; j++){
          sum += dz(i,j)*last_input(stride*i+k-padding, stride*j+l-padding);
        }
      }
      dK(k,l) = sum;
    }
  }

  double db = std::accumulate(dz.data.begin(), dz.data.end(), 0.0);

  Matrix dx(last_input.rows,last_input.cols);
  for (int i{0}; i < last_input.rows; i++){
    for (int j{0}; j < last_input.cols; j++){
      double sum = 0.0;
      for (int k{0}; k < kernel_size; k++){
        for (int l{0}; l < kernel_size; l++){
          sum += dz((i+padding-k)/stride, (j+padding-l)/stride)*kernel(k,l);
        }
      }
      dx(i,j) = sum;
    }
  }

  grad_kernel = dK;
  grad_bias = db;

  return dx;
}

Matrix MaxPoolLayer::forward(const Matrix& input) {
  last_input = input;
  return input.max_pooling(pool_size, stride, padding);
}

Matrix MaxPoolLayer::backward(const Matrix& grad_output){
  Matrix dx(last_input.rows, last_input.cols);

  int out_rows = grad_output.rows;
  int out_cols = grad_output.cols;

  for (int i{0}; i < out_rows; i++){
    for (int j{0}; j < out_cols; j++){
      double max_val = -std::numeric_limits<double>::infinity();
      int max_r = -1, max_c = 1;

      for (int p{0}; p < pool_size; p++){
        for (int q{0}; q < pool_size; q++){
          int r = i*stride + p -padding;
          int c = j*stride + q -padding;
          if (r < 0 || r >=last_input.rows || c < 0 || c >= last_input.cols) continue;
          if (last_input(r,c) > max_val) {
            max_val = last_input(r,c);
            max_r = r;
            max_c = c;
          }
        }
      }
      if (max_r != -1 && max_c != -1) {
        dx(max_r, max_c) += grad_output(i,j);
      }
    }
  }
  return dx;
}

Matrix MeanPoolLayer::forward(const Matrix& input){
  last_input = input;
  return input.mean_pooling(pool_size, stride, padding);
}

Matrix MeanPoolLayer::backward(const Matrix& grad_output) {
  Matrix dx(last_input.rows, last_input.cols);
  int div = pool_size * pool_size;

  int out_rows = grad_output.rows;
  int out_cols = grad_output.cols;

  for (int i{0}; i < out_rows; i++){
    for (int j{0}; j < out_cols; j++){
      double mean = grad_output(i,j) / div;

      for (int p{0}; p < pool_size; p++){
        for (int q{0}; q < pool_size; q++){
          int r = i*stride + p - padding;
          int c = j*stride + q - padding;
          if (r < 0 || r >= last_input.rows || c < 0 || c >= last_input.cols) continue;
          dx(r,c) += mean;
        }
      }
    }
  }
  return dx;
}
