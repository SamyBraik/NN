#ifndef NN_H
#define NN_H

#include "algebra.h"
#include "activation.h"
#include <string>
#include <random>
#include "loss.h"
#include "optimizer.h"

class Layer {
  public :
    Layer(int n_in, int n_out, bool output_layer = false);

    Vector forward(const Vector& input);
    Vector backward(const Vector& grad_output);
    void update_weights(Optimizer& optimizer);

    int n_in, n_out;
    bool output_layer;
    
    Matrix weights;
    Vector bias;
    
    Matrix grad_weights;
    Vector grad_bias;

    Vector last_input, last_z;

  private:
    void randomized_weights();
};

class MLP {
  public :
    MLP(std::vector<int> layer_size, OptimizerType opt_type, double learning_rate);

    Vector forward(const Vector& input);
    double compute_loss(const Vector& prediction, const Vector& target, std::string loss) const;
    void backward(const Vector& prediction, const Vector& target);
    void update_weights();
    std::vector<Layer> layers;
    std::vector<Optimizer*> optimizers;
};


class ConvLayer {
  public : 
    ConvLayer(int kernel_size, int stride)
      : kernel(kernel_size, kernel_size), stride(stride), bias(0.0), grad_kernel(kernel_size, kernel_size), 
      grad_bias(0.0), last_input(1,1), last_z(1,1) {randomized_kernel();}

    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& grad_output);

    int kernel_size, stride;
    Matrix kernel;
    double bias;

    Matrix grad_kernel;
    double grad_bias;

    Matrix last_input, last_z;
 

  private : 
    void randomized_kernel();
};

#endif
