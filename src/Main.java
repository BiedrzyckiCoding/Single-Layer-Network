import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

import SingleLayerNetwork.*;

public class Main {
    //codes for coloring (taken from stackoverflow)
    public static final String ANSI_GREEN = "\u001B[32m";
    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        String trainFile = "resources/lang.train.csv";
        String testFile = "resources/lang.test.csv";
        double learningRate = 0.1;
        int epochs = 20;

        try {
            //build & train
            SingleLayerNetwork net = new SingleLayerNetwork(trainFile, testFile, learningRate, epochs);
            net.train();

            //color-coded test output
            System.out.println("\n=== Test set results ===");
            BufferedReader br = new BufferedReader(new FileReader(testFile));
            String line;
            int total = 0, correct = 0;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                //split on first comma -> [trueLabel, rawText]
                String[] parts = line.split(",", 2);
                String trueLabel = parts[0];
                String rawText = (parts.length > 1 ? parts[1] : "");
                String predLabel = net.classify(rawText);

                boolean isCorrect = predLabel.equals(trueLabel);
                if (isCorrect) correct++;
                total++;

                String color = isCorrect ? ANSI_GREEN : ANSI_RED;
                //print entire original line + arrow + predicted label
                System.out.println(color + line + "  ->  " + predLabel + ANSI_RESET);
            }
            br.close();

            double accuracy = 100.0 * correct / total;
            System.out.printf("%nOverall accuracy: %.2f%%%n", accuracy);

            //user input
            Scanner sc = new Scanner(System.in);
            System.out.println("\nEnter your own text to classify (empty line to quit):");
            while (true) {
                System.out.print("> ");
                String userText = sc.nextLine().trim();
                if (userText.isEmpty()) break;
                String lang = net.classify(userText);
                System.out.println("Detected language: " + lang);
            }
            sc.close();

        } catch (IOException ex) {
            System.err.println("Error reading data files: " + ex.getMessage());
            System.exit(1);
        }
    }
}