package NeuralNetwork;

import java.util.function.Function;

public class Squisher {
    private Function<Double, Double> squish;
    private Function<Double, Double> prime;
    public Squisher(Function<Double, Double> squish, Function<Double, Double> prime) {
        this.squish = squish;
        this.prime = prime;
    }
    public Function<Double, Double> squish() {
        return squish;
    }
    public Function<Double, Double> prime() {
        return prime;
    }
}
