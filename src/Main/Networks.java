package Main;

import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.Squisher;
import NeuralNetwork.SquishFunctions;
import java.util.Arrays;

import java.util.Scanner;
import java.util.function.Function;
import java.util.stream.Stream;

public class Networks {
    private static final Scanner in = new Scanner(System.in);
    public static boolean AutoSave_Network;
    public static boolean AutoSave_Results;

    public static final int Command_Ind = 0;
    public static final int Network_Ind = 1;
    public static final String Settings = "settings";
    public static final String New = "new";
    public static final String Train = "train";
    public static final String Quit = "quit";

    public static void Display_Settings() {
        String output = "";
        output += "\nAutoSave_Network: " + (AutoSave_Network ? "On" : "Off");
        output += "\nAutoSave_Results: " + (AutoSave_Results ? "On" : "Off");

        System.out.println(output);
    }

//    public static void Train(String net) {
//        NeuralNetwork network = ()
//    }

    public static NeuralNetwork new_Network() {
        int[] sizes = get_sizes();
        String s = in.nextLine();
        double rate = Double.parseDouble(s);
        Squisher squisher = getSquisher();
        return new NeuralNetwork(sizes, squisher, rate);
    }

    private static int[] get_sizes() {
        System.out.print("Enter layer sizes:");
        String[] ss = in.nextLine().split(",");
        int[] sizes = new int[ss.length];
        for (int i = 0; i < ss.length; i++) {
            sizes[i] = Integer.parseInt(ss[i]);
        }
        return sizes;
    }

    public static Squisher getSquisher() {
        String string = in.nextLine();
        Squisher squisher;
        switch (string) {
            case "sigmoid": squisher = SquishFunctions.sigmoid_squish;
            case "relu":
            case "rectified linear units": squisher = SquishFunctions.relu_squish;
            default:
                System.out.println("Function not found");
                squisher = getSquisher();
        }
        return squisher;
    }

    public static void List_Commands() {
        String output = "";
        output += Settings + ": displays program settings\n";
        output += Quit + ": quit program\n";

        System.out.println(output);
    }

    public static void main(String[] args) {

        boolean done = false;
        NeuralNetwork network;
        while (!done) {
            String command = args[Command_Ind];
            switch (command) {
                case Settings: Display_Settings();
//            case Train: Train();
                case New: network = new_Network();
                case Quit: done = true;
                default: List_Commands();
            }
            args = in.nextLine().split(" ");
        }
        if (AutoSave_Network) {

        }
    }
}
