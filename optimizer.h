#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "algebra.h"

class Optimizer {
  public :
    virtual void update(Matrix& weights, Matrix& grad_weights, Vector& bias, Vector& grad_bias) = 0;
    virtual ~Optimizer() = default;
};

class SGD : public Optimizer {
  public :
    SGD(double learning_rate = 0.01) : learning_rate(learning_rate) {}
    
    void update(Matrix& weights, Matrix& grad_weights, Vector& bias, Vector& grad_bias) override {
      weights = weights + grad_weights *-learning_rate;
      bias = bias + grad_bias * -learning_rate;
    }

  private : 
    double learning_rate;
};

class ADAM : public Optimizer {
  public :
    ADAM(int rows, int cols, double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.9, double eps = 1e-8) :
      m_w(rows, cols), v_w(rows, cols), m_b(rows), v_b(rows), alpha(alpha), beta1(beta1), beta2(beta2), eps(eps), t(0) {}

    void update(Matrix& weights, Matrix& grad_weights, Vector& bias, Vector& grad_bias) override {
      t++;

      m_w = m_w*beta1 + grad_weights*(1-beta1);
      m_b = m_b*beta1 + grad_bias*(1-beta1);

      v_w = v_w*beta2 + grad_weights.hadamard(grad_weights)*(1-beta2);
      v_b = v_b*beta2 + grad_bias.hadamard(grad_bias)*(1-beta2);

      Matrix m_w_esti = m_w / (1-std::pow(beta1,t));
      Vector m_b_esti = m_b / (1-std::pow(beta1,t));

      Matrix v_w_esti = v_w / (1-std::pow(beta2,t));
      Vector v_b_esti = v_b / (1-std::pow(beta2,t));

      weights = weights - (m_w_esti / (v_w_esti.power(0.5) + eps))*alpha;
      bias = bias - (m_b_esti / (v_b_esti.power(0.5) + eps))*alpha;
    }

  private :
    Matrix m_w, v_w;
    Vector m_b, v_b;
    double alpha, beta1, beta2, eps;
    int t;
};

enum class OptimizerType {SGD_,ADAM_};

inline Optimizer* make_optimizer(OptimizerType type, int rows, int cols, double learning_rate){
  switch (type) {
    case OptimizerType::SGD_ : return new SGD(learning_rate);
    case OptimizerType::ADAM_ : return new ADAM(rows, cols, learning_rate);
  }
  return nullptr;
}

#endif 
