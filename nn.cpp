#include "nn.h"
#include <random>
#include <stdexcept>
#include "optimizer.h"
#include <iostream>

Layer::Layer(int n_in, int n_out, bool output_layer) : n_in(n_in), n_out(n_out), output_layer(output_layer), weights(n_out, n_in), bias(n_out), last_input(n_in), last_z(n_out), grad_weights(n_out, n_in), grad_bias(n_out) {randomized_weights();}

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
  for (int i{0}; i < layer_size.size()-1; i++){
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

  for (int i{0}; i < layers.size(); i++){
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
  for (int i{0}; i < layers.size(); i++){
    layers[i].update_weights(*optimizers[i]);
  }
}

