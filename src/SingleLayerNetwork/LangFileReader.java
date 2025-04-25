package SingleLayerNetwork;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;

public class LangFileReader {
    public static class DataSet {
        public ArrayList<ArrayList<Double>> X;
        public ArrayList<Integer> y;
        public ArrayList<String> rawText;    // for misclassification display
        public ArrayList<String> labelNames; // maps index→language
    }

    public static DataSet read(String path) throws IOException {
        BufferedReader br = new BufferedReader(new java.io.FileReader(path));
        String line;
        ArrayList<String> texts = new ArrayList<>();
        ArrayList<String> labels = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;
            String[] p = SingleLayerNetwork.safeSplitTextLabel(line);
            texts.add(p[0]);
            labels.add(p[1]);
        }
        // build a label→index map in order of appearance
        LinkedHashMap<String,Integer> map = new LinkedHashMap<>();
        for (String lbl : labels) {
            if (!map.containsKey(lbl)) {
                map.put(lbl, map.size());
            }
        }
        // vectorize
        DataSet ds = new DataSet();
        ds.X = new ArrayList<>();
        ds.y = new ArrayList<>();
        ds.rawText = new ArrayList<>();
        ds.labelNames = new ArrayList<>(map.keySet());
        for (int i = 0; i < texts.size(); i++) {
            ds.X.add(TextVectorizer.vectorize(texts.get(i)));
            ds.y.add(map.get(labels.get(i)));
            ds.rawText.add(texts.get(i));
        }
        return ds;
    }
}
