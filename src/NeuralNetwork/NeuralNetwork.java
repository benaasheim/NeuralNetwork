package NeuralNetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

public class NeuralNetwork {
    private ArrayList<double[][]> weights;
    private ArrayList<double[]> biases;
    private double learningRate;
    private Function<Double, Double> squish;
    private Function<Double, Double> squish_prime;
    private BiFunction<Double, Double, Double> mlt = (a, b) -> a*b;
    private BiFunction<Double, Double, Double> add = (a, b) -> a+b;
    private BiFunction<Double, Double, Double> sub = (a, b) -> a-b;

    public NeuralNetwork(int[] sizes, Function<Double, Double> squish, Function<Double, Double> squish_prime, double rate) {
        weights = new ArrayList<>();
        biases = new ArrayList<>();
        for (int i = 1; i < sizes.length; i++) {

            // the number of neurons in the current layer
            int n1 = sizes[i];

            // the number of neurons in the previous layer
            int n0 = sizes[i-1];

            // add weight layer to weights
            double[][] weight_layer = new double[n1][n0];
            weights.add(weight_layer);

            // add bias layer to biases
            double[] bias_layer = new double[n1];
            biases.add(bias_layer);
        }
        this.squish = squish;
        this.squish_prime = squish_prime;
        learningRate = rate;
    }
    public NeuralNetwork(int[] sizes, double rate) {
        weights = new ArrayList<>();
        biases = new ArrayList<>();
        for (int i = 1; i < sizes.length; i++) {

            // the number of neurons in the current layer
            int n1 = sizes[i];

            // the number of neurons in the previous layer
            int n0 = sizes[i-1];

            // add weight layer to weights
            double[][] weight_layer = new double[n1][n0];
            weights.add(weight_layer);

            // add bias layer to biases
            double[] bias_layer = new double[n1];
            biases.add(bias_layer);
        }
        learningRate = rate;
        squish = SquishFunctions.sigmoid;
        squish_prime = SquishFunctions.sigmoid_prime;
    }
    public NeuralNetwork(int[] sizes, Squisher squisher, double rate) {
        weights = new ArrayList<>();
        biases = new ArrayList<>();
        for (int i = 1; i < sizes.length; i++) {

            // the number of neurons in the current layer
            int n1 = sizes[i];

            // the number of neurons in the previous layer
            int n0 = sizes[i-1];

            // add weight layer to weights
            double[][] weight_layer = new double[n1][n0];
            weights.add(weight_layer);

            // add bias layer to biases
            double[] bias_layer = new double[n1];
            biases.add(bias_layer);
        }
        learningRate = rate;
        squish = squisher.squish();
        squish_prime = squisher.prime();
    }
    public NeuralNetwork(int[] sizes, Squisher squisher) {
        weights = new ArrayList<>();
        biases = new ArrayList<>();
        for (int i = 1; i < sizes.length; i++) {

            // the number of neurons in the current layer
            int n1 = sizes[i];

            // the number of neurons in the previous layer
            int n0 = sizes[i-1];

            // add weight layer to weights
            double[][] weight_layer = new double[n1][n0];
            weights.add(weight_layer);

            // add bias layer to biases
            double[] bias_layer = new double[n1];
            biases.add(bias_layer);
        }
        learningRate = 0.1;
        squish = squisher.squish();
        squish_prime = squisher.prime();
    }
    private void trainXloops(double[] input, double[] expected, int loops) {
        for (int i = 0; i < loops; i++) {
            System.out.println(Arrays.toString(trainingLoop(input, expected)));
        }
    }
    private double[] squish(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = squish.apply(input[i]);
        }
        return output;
    }
    private double[] squish_prime(double[] input) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = squish_prime.apply(input[i]);
        }
        return output;
    }
    private double[] trainingLoop(double[] input, double[] expected) {
        return propForward(input, expected, 0);
    }
    private double[] propForward(double[] input, double[] expected, int layer) {
        //weights of current layer
        double[][] WL = weights.get(layer);

        //biases of current layer
        double[] BL = biases.get(layer);

        //activation output of previous layer
        double[] ALn1 = input;

        // z = w*a[L-1] + b
//        double[] ZL = add(dot(WL, ALn1), BL);
        double[] ZL = func1t1(dot(WL, ALn1), BL, add);

        //Activation of Current layer = squish_function(zL)
        double[] AL = squish(ZL);

        double[] dCostdAL;
        if (layer == (biases.size() -1)) {
            //the derivative of the Cost with respect to the Current Layer's Activation = 2*(AL-Y)
            dCostdAL = Cost(AL, expected);
        }
        else {
            dCostdAL = propForward(AL, expected, layer+1);
        }

        //the derivative of the Current Layer's Activation with respect to the pre-squished value
        double[] dALdZL = squish_prime(ZL);

        /*
        The Derivative of the cost function with respect to the biases equals
        dCostdBL =
        the derivative of the Cost with respect to the Current Layer's Activation times
        dCostdAL *
        the derivative of the Current Layer's Activation with respect to the pre-squished value
        dALdZL
         */
        double[] dCostdBL = func1t1(dCostdAL, dALdZL, mlt);

        /*
        The Derivative of the cost function with respect to the weights equals
        dCostdWL =
        the derivative of the Cost with respect to the Current Layer's Activation times
        dCostdAL *
        the derivative of the Current Layer's Activation with respect to the pre-squished value
        dALdZL *
        The Derivative of the pre-squished value with respect to the weights
        dZLdWL

        Since the first two operands have already been combined into dCostdBL, we can re-write as
        dCostdWL = dCostdBL * dZLdWL
         */
        double[] dZLdWL = ALn1;
        double[][] dCostdWL = dot(dCostdBL, dZLdWL);

        // Scale the changes by the learning rate and alter
        WL = func2t2(WL, func2t0(dCostdWL, learningRate, mlt), sub);
        BL = func1t1(BL, func1t0(dCostdBL, learningRate, mlt), sub);
        weights.set(layer, WL);
        biases.set(layer, BL);

        /*
        Finally we return the The Derivative of the cost function,
        with respect to the Previous Layer's Activation
        to be used in the last recursion call's equations

        The Derivative of the cost function with respect to the Previous Layer's Activation equals
        dCostdA(L-1) =
        the derivative of the Cost with respect to the Current Layer's Activation times
        dCostdAL *
        the derivative of the Current Layer's Activation with respect to the pre-squished value
        dALdZL *
        The Derivative of the pre-squished value with respect to the weights
        dZLdA(L-1)

        Since the first two operands have already been combined into dCostdBL, we can re-write as
        dCostdA(L-1) = dCostdBL * dZLdA(L-1)

        variable dZLdA(L-1) will be written as dZLdALn1
         */
        double[] dCostdALn1 = dot(dALdZL, WL);
        return dCostdALn1;
    }

    private double[] Cost(double[] input, double[] expected) {
        System.out.println(Arrays.toString(input));
        return  func1t0(func1t1(input, expected, sub), 2.0, mlt);
    }

    private double[] dot(double[][] A, double[] B) {
        double[] C = new double[A.length];
        for (int i = 0; i < A.length; i++) {
            double sum = 0;
            for (int j = 0; j < B.length; j++) {
                sum += A[i][j] * B[j];
            }
            C[i] = sum;
        }
        return C;
    }
    private double[] dot(double[] B, double[][] A) {
        double[] C = new double[A[0].length];
        for (int i = 0; i < A[0].length; i++) {
            double sum = 0;
            for (int j = 0; j < A.length; j++) {
                sum += A[j][i] * B[j];
            }
            C[i] = sum;
        }
        return C;
    }
    private double[][] dot(double[] A, double[] B) {
        double[][] C = new double[A.length][B.length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < B.length; j++) {
                C[i][j] = A[i] + B[j];
            }
        }
        return C;
    }
    private double[][] func2t0(double[][] A, double B, BiFunction<Double, Double, Double> function) {
        double[][] C = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                C[i][j] = function.apply(A[i][j], B);
            }
        }
        return C;
    }

    private double[][] func2t2(double[][] A, double B[][], BiFunction<Double, Double, Double> function) {
        double[][] C = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            C[i] = func1t1(A[i], B[i], function);
        }
        return C;
    }

    private double[] func1t1(double[] A, double[] B, BiFunction<Double, Double, Double> function) {
        double[] C = new double[A.length];
        for (int i = 0; i < A.length; i++) {
            C[i] = function.apply(A[i], B[i]);
        }
        return C;
    }

    private double[] func1t0(double[] A, double B, BiFunction<Double, Double, Double> function) {
        double[] C = new double[A.length];
        for (int i = 0; i < A.length; i++) {
            C[i] = function.apply(A[i], B);
        }
        return C;
    }

    public static void main(String[] args) {
        double[] X = {0, 1, 1, 0};
        double[] Y = {0, 0, 1, 0};

        int loops = 1000;

        int[] sizes = {4, 4, 4, 8, 4};
        NeuralNetwork neuralNetwork = new NeuralNetwork(sizes, 0.1);
        neuralNetwork.trainXloops(X, Y, loops);
    }
}
