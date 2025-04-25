import SingleLayerNetwork.LangFileReader;
import SingleLayerNetwork.SingleLayerNetwork;
import SingleLayerNetwork.TextVectorizer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class Main {
    // safeSplitTextLabel and everything above goes here

    public static void main(String[] args) throws IOException {
        // 1. load train & test
        var train = LangFileReader.read("resources/lang.train.csv");
        var test  = LangFileReader.read("resources/lang.test.csv");

        int dim = train.X.get(0).size();
        double eta = 0.01;
        SingleLayerNetwork net = new SingleLayerNetwork(dim, train.labelNames, eta);

        // 2. train
        int epochs = 10;
        net.train(train.X, train.y, epochs);

        // 3. evaluate
        int correct = 0;
        System.out.println("Misclassified examples:");
        for (int i = 0; i < test.X.size(); i++) {
            String pred = net.predictLabel(test.X.get(i));
            String truth = test.labelNames.get(test.y.get(i));
            if (pred.equals(truth)) {
                correct++;
            } else {
                System.out.printf(" > “%s” → %s (should be %s)%n",
                        test.rawText.get(i), pred, truth);
            }
        }
        double acc = 100.0 * correct / test.X.size();
        System.out.printf("Test accuracy: %.2f%%%n", acc);

        // 4. CLI for new inputs
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter any text to classify (blank to quit):");
        while (true) {
            String line = sc.nextLine();
            if (line.trim().isEmpty()) break;
            var vec = TextVectorizer.vectorize(line);
            String lang = net.predictLabel(vec);
            System.out.println("→ “" + lang + "”");
        }
        sc.close();
    }
}
