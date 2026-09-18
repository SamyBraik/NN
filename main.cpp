#include <iostream>
#include "nn.h"

int main(){
  MLP net({2,4,8,16,4,1});
  double learning_rate = 0.1;
  int epochs = 1000;
  
  std::vector<Vector> inputs;
  std::vector<Vector> targets;

  auto make_pair = [](double a, double b, double y) {
    Vector in(2); in(0) = a; in(1) = b;
    Vector out(1); out(0) = y;
    return std::make_pair(in, out);
  };

  std::vector<std::pair<Vector,Vector>> data = {
    make_pair(0,0, 0),
    make_pair(0,1, 1),
    make_pair(1,0, 1),
    make_pair(1,1, 0),
  };

  for (int epoch{0}; epoch < epochs; epoch++){
    double total_loss = 0.0;

    for (auto& [input, target] : data){
      Vector prediction = net.forward(input);
      total_loss += net.compute_loss(prediction, target, "MSE");
 
      net.backward(prediction, target);
      net.update_weights(learning_rate);
    }

    if (epoch % 100 == 0) std::cout<< "Epoch :" << epoch << ", loss :" << total_loss << '\n';
  }

  return 0;
}
