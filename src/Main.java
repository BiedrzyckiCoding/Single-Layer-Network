import java.util.ArrayList;
import java.util.Random;

public class Main {
    public static void main(String[] args) {

        //read a training set and a test set using FileReader methods.
        FileReader.DataSet trainData = FileReader.readCSV("resources/perceptron.data");
        FileReader.DataSet testData  = FileReader.readCSV("resources/perceptron.test.data");

        //initialize a Perceptron
        //automatically set the number of weights to match the dimension of the training data's feature vectors.
        int dimension = trainData.data.get(0).size();
        ArrayList<Double> initialWeights = new ArrayList<>();
        Random rand = new Random();
        for (int i = 0; i < dimension; i++) {
            initialWeights.add(rand.nextDouble()); //random [0,1]
        }
        System.out.println("initial weights: " + initialWeights);
        double initialBias = rand.nextDouble(); //random [0,1]
        double learningRate = 0.01; //small learning rate

        Perceptron perceptron = new Perceptron(initialWeights, initialBias, learningRate);

        //train the perceptron on the training data
        int epochs = 10;
        perceptron.train(trainData.data, trainData.labels, epochs);

        //evaluate on the test set
        int correctCount = 0;
        for (int i = 0; i < testData.data.size(); i++) {
            ArrayList<Double> x = testData.data.get(i);
            int label = testData.labels.get(i);
            int prediction = perceptron.output(x);
            if (prediction == label) {
                correctCount++;
            }
        }
        double accuracy = 100.0 * correctCount / testData.data.size();
        System.out.println("Accuracy on test set: " + accuracy + "%");

        //Gui gui = new Gui();
    }
}
