package Perceptron;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;

public class FileReader {
    public static class DataSet {
        public ArrayList<ArrayList<Double>> data;
        public ArrayList<Integer> labels;

        public DataSet(ArrayList<ArrayList<Double>> data, ArrayList<Integer> labels) {
            this.data = data;
            this.labels = labels;
        }
    }
    public static DataSet readCSV(String filePath) {
        ArrayList<ArrayList<Double>> data = new ArrayList<>();
        ArrayList<Integer> labels = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new java.io.FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue; // skip empty lines
                }
                String[] parts = line.split(",");
                //the last column is the label
                String labelStr = parts[parts.length - 1];

                //convert the label to integer if recognized, or parse it
                int label;
                if (labelStr.equalsIgnoreCase("Iris-virginica")) {
                    label = 1;
                } else if (labelStr.equalsIgnoreCase("Iris-versicolor")) {
                    label = 0;
                } else {
                    label = Integer.parseInt(labelStr);
                }

                //the remaining columns (all but last) are numeric features
                ArrayList<Double> row = new ArrayList<>();
                for (int i = 0; i < parts.length - 1; i++) {
                    row.add(Double.parseDouble(parts[i]));
                }
                data.add(row);
                labels.add(label);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (NumberFormatException nfe) {
            System.err.println("Error parsing a numeric value in " + filePath + ": " + nfe.getMessage());
        }

        return new DataSet(data, labels);
    }
}
