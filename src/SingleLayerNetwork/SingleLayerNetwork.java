package SingleLayerNetwork;

import Perceptron.*;

import java.io.*;
import java.io.FileReader;
import java.util.*;

public class SingleLayerNetwork {
    private final List<Perceptron> perceptrons = new ArrayList<>();
    private final Map<String,Integer> labelToIndex = new HashMap<>();
    private final List<String> indexToLabel = new ArrayList<>();

    private final List<ArrayList<Double>> trainData = new ArrayList<>();
    private final List<Integer> trainLabels = new ArrayList<>();
    private final List<ArrayList<Double>> testData = new ArrayList<>();
    private final List<Integer> testLabels = new ArrayList<>();

    private final double learningRate;
    private final int epochs;

    public SingleLayerNetwork(String trainPath, String testPath, double learningRate, int epochs) throws IOException {
        this.learningRate = learningRate;
        this.epochs= epochs;

        //load train -> builds labelToIndex on the fly
        loadData(trainPath, trainData, trainLabels);

        //init one perceptron per discovered language
        initializePerceptrons();

        //load test (labels must already be known)
        loadData(testPath, testData, testLabels);
    }

    private void loadData(String path, List<ArrayList<Double>> data, List<Integer> labels) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                //split only on first comma
                String[] parts = line.split(",", 2);
                String label = parts[0];
                String text = parts.length > 1 ? parts[1] : "";

                //register new label if in training
                if (!labelToIndex.containsKey(label) && data == trainData) {
                    labelToIndex.put(label, indexToLabel.size());
                    indexToLabel.add(label);
                }

                Integer idx = labelToIndex.get(label);
                if (idx == null) {
                    //unseen label in test set -> skip or error
                    System.err.println("Unknown label in test: " + label);
                    continue;
                }

                data.add(vectorize(text));
                labels.add(idx);
            }
        }
    }

    private ArrayList<Double> vectorize(String text) {
        double[] counts = new double[26];
        text = text.toLowerCase();
        for (char c : text.toCharArray()) {
            if (c >= 'a' && c <= 'z') {
                counts[c - 'a']++;
            }
        }
        double norm = 0;
        for (double v : counts) norm += v*v;
        norm = Math.sqrt(norm);
        ArrayList<Double> vec = new ArrayList<>(26);
        for (int i = 0; i < 26; i++) {
            vec.add(norm > 0 ? counts[i]/norm : 0.0);
        }
        return vec;
    }

    private void initializePerceptrons() {
        for (int i = 0; i < indexToLabel.size(); i++) {
            //zero weights, zero bias
            ArrayList<Double> w = new ArrayList<>(Collections.nCopies(26, 0.0));
            perceptrons.add(new Perceptron(w, 0.0, learningRate));
        }
    }

    public void train() {
        System.out.println("Training for up to " + epochs + " epochs...");
        for (int epoch = 0; epoch < epochs; epoch++) {
            int errors = 0;
            for (int i = 0; i < trainData.size(); i++) {
                ArrayList<Double> x       = trainData.get(i);
                int               trueIdx = trainLabels.get(i);

                //one-vs-all update
                for (int j = 0; j < perceptrons.size(); j++) {
                    Perceptron p = perceptrons.get(j);
                    int target    = (j == trueIdx) ? 1 : 0;
                    int pred      = p.output(x);
                    if (pred != target) {
                        p.updateWeights(x, target, pred);
                        errors++;
                    }
                }
            }
            System.out.printf(" Epoch %2d — errors = %d%n", epoch, errors);
            if (errors == 0) {
                System.out.println("Converged early at epoch " + epoch);
                break;
            }
        }
    }

/*    public double evaluate() {
        int correct = 0;
        for (int i = 0; i < testData.size(); i++) {
            ArrayList<Double> x = testData.get(i);
            int trueIdx = testLabels.get(i);

            //pick perceptron with largest raw net
            double bestNet = Double.NEGATIVE_INFINITY;
            int bestJ   = -1;
            for (int j = 0; j < perceptrons.size(); j++) {
                Perceptron p = perceptrons.get(j);
                //compute w·x - bias
                double sum = 0;
                for (int k = 0; k < 26; k++) {
                    sum += x.get(k) * p.weights.get(k);
                }
                double net = sum - p.bias;
                if (net > bestNet) {
                    bestNet = net;
                    bestJ   = j;
                }
            }
            if (bestJ == trueIdx) correct++;
        }
        double acc = 100.0 * correct / testData.size();
        System.out.printf("Test accuracy: %.2f%%%n", acc);
        return acc;
    }*/

    public String classify(String text) {
        ArrayList<Double> x = vectorize(text);

        double bestNet = Double.NEGATIVE_INFINITY;
        int bestJ = -1;
        for (int j = 0; j < perceptrons.size(); j++) {
            Perceptron p = perceptrons.get(j);
            double sum = 0;
            for (int k = 0; k < 26; k++) sum += x.get(k) * p.weights.get(k);
            double net = sum - p.bias;
            if (net > bestNet) {
                bestNet = net;
                bestJ = j;
            }
        }
        return indexToLabel.get(bestJ);
    }
}
