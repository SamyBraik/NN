#ifndef NN_H
#define NN_H

#include "algebra.h"
#include "activation.h"

class Layer {
  public :
    Layer(int n_in, int n_out);

    Vector forward(const Vector& input);
    Vector backward(const Vector& grad_output);
    void update_weights(double learning_rate);

    int n_in, n_out;
    
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
    MLP(std::vector<int> layer_size);

    Vector forward(const Vector& input);
    double compute_loss(const Vector& prediction, const Vector& target) const;
    void backward(const Vector& prediction, const Vector& target);
    void update_weights(double learning_rate);

    std::vector<Layer> layers;
};

#endif
