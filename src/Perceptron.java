import java.util.ArrayList;

public class Perceptron {
    ArrayList<Double> inputVec;
    ArrayList<Double> weights;
    double bias;
    double learningRate;
    double expectedOutput;
    public String testFilePath;
    public String trainFilePath;


    public int netVal(double input){
        int result = 0;
        if(input >= 0){
            result = 1;
        }
        return result;
    }

    public Perceptron(ArrayList<Double> weights, double bias, double learningRate) {
        this.weights = weights;
        this.bias = bias;
        this.learningRate = learningRate;
    }

    public int output(ArrayList<Double> inputVec) {
        double sum = 0.0;
        for (int i = 0; i < inputVec.size(); i++) {
            sum += inputVec.get(i) * weights.get(i);
        }
        //net input = (w . x) - bias
        double net = sum - bias;
        //pass net through activation function
        return netVal(net);
    }

    private void updateWeights(ArrayList<Double> xVector, int label, int predicted) {
        double constant = learningRate * (label - predicted);
        //update each weight
        for (int i = 0; i < weights.size(); i++) {
            double newWeight = weights.get(i) + (xVector.get(i) * constant);
            weights.set(i, newWeight);
        }
        //update bias
        bias = bias - constant;
    }

    public void train(ArrayList<ArrayList<Double>> data, ArrayList<Integer> labels, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            int errorCount = 0;
            for (int i = 0; i < data.size(); i++) {
                ArrayList<Double> xVector = data.get(i);
                int label = labels.get(i);
                int prediction = output(xVector);

                //if misclassified, update weights
                if (prediction != label) {
                    updateWeights(xVector, label, prediction);
                    errorCount++;
                }
            }
            //if no errors were found, we've converged early
            if (errorCount == 0) {
                System.out.println("Converged after " + epoch + " epochs.");
                break;
            }
        }
    }
}
