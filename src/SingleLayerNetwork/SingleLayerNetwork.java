package SingleLayerNetwork;

import Perceptron.Perceptron;
import java.util.ArrayList;
import java.util.Random;

public class SingleLayerNetwork {
    private final ArrayList<Perceptron> perceptrons;
    private final ArrayList<String> labelNames;

    public SingleLayerNetwork(int inputDim, ArrayList<String> labelNames, double learningRate) {
        this.labelNames = labelNames;
        this.perceptrons = new ArrayList<>();
        Random rnd = new Random();
        for (int i = 0; i < labelNames.size(); i++) {
            ArrayList<Double> w = new ArrayList<>();
            for (int j = 0; j < inputDim; j++) {
                w.add(rnd.nextDouble());
            }
            double b = rnd.nextDouble();
            perceptrons.add(new Perceptron(w, b, learningRate));
        }
    }

    //helper to split the text safely
    public static String[] safeSplitTextLabel(String line) {
        // split only on the last comma
        int idx = line.lastIndexOf(',');
        String text = line.substring(0, idx);
        String label = line.substring(idx + 1);
        return new String[]{ text, label };
    }


    /** train one‐vs‐all for each epoch */
    public void train(ArrayList<ArrayList<Double>> X, ArrayList<Integer> y, int epochs) {
        for (int e = 0; e < epochs; e++) {
            int errors = 0;
            for (int i = 0; i < X.size(); i++) {
                ArrayList<Double> xi = X.get(i);
                int yi = y.get(i);
                // for each perceptron k, target = (k == yi ? 1 : 0)
                for (int k = 0; k < perceptrons.size(); k++) {
                    int pred = perceptrons.get(k).output(xi);
                    int target = (k == yi ? 1 : 0);
                    if (pred != target) {
                        perceptrons.get(k).updateWeights(xi, target, pred);
                        errors++;
                    }
                }
            }
            if (errors == 0) {
                System.out.println("Converged after " + e + " epochs.");
                break;
            }
        }
    }

    /** pick the perceptron with the largest linear net input */
    public int predictIndex(ArrayList<Double> x) {
        double best = Double.NEGATIVE_INFINITY;
        int argmax = -1;
        for (int k = 0; k < perceptrons.size(); k++) {
            Perceptron p = perceptrons.get(k);
            // replicate net = w·x – bias
            double sum = 0;
            for (int j = 0; j < x.size(); j++) {
                sum += x.get(j)*p.weights.get(j);
            }
            double net = sum - p.bias;
            if (net > best) {
                best = net;
                argmax = k;
            }
        }
        return argmax;
    }

    public String predictLabel(ArrayList<Double> x) {
        return labelNames.get(predictIndex(x));
    }
}
