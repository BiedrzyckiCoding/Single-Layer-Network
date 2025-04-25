package SingleLayerNetwork;

import java.util.ArrayList;

public class TextVectorizer {
    public static ArrayList<Double> vectorize(String text) {
        double[] counts = new double[26];
        for (char c : text.toLowerCase().toCharArray()) {
            if (c >= 'a' && c <= 'z') {
                counts[c - 'a']++;
            }
        }
        // norm
        double norm = 0;
        for (double v : counts) norm += v*v;
        norm = Math.sqrt(norm);
        ArrayList<Double> v = new ArrayList<>(26);
        for (double x : counts) {
            v.add(norm > 0 ? x / norm : 0.0);
        }
        return v;
    }
}
