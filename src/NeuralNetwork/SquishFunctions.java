package NeuralNetwork;

import java.util.function.Function;

public interface SquishFunctions {
    Function<Double, Double> sigmoid = a -> (1.0 / (1 + Math.exp(-a)));
    Function<Double, Double> sigmoid_prime = a -> sigmoid.apply(a)*(1-(sigmoid.apply(a)));
    Squisher sigmoid_squish = new Squisher(sigmoid, sigmoid_prime);
    Function<Double, Double> relu = a -> a < 0.0 ? 0.0 : a;
    Function<Double, Double> relu_prime = a -> a < 0.0 ? 0.0 : 1.0;
    Squisher relu_squish = new Squisher(relu, relu_prime);
}
