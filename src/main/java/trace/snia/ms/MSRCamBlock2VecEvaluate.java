package trace.snia.ms;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeSet;

/**
 * Created by daidong on 3/2/16.
 */
public class MSRCamBlock2VecEvaluate {

    public static final int window = 3;
    public static final int number = 30;
    public static final int block_size = 4096;

    public static void main(String args[]) throws IOException {

        String trainedModel = args[0];
        String verifyFile = args[1];
        String outputFile = args[2];

        BufferedReader brModel = new BufferedReader(new FileReader(trainedModel));
        String line;
        String firstLen = brModel.readLine();
        int vocab_size = Integer.parseInt(firstLen.split(" ")[0]);
        int vec_size = Integer.parseInt(firstLen.split(" ")[1]);
        double[][] vectors = new double[vocab_size][vec_size];
        HashMap<ByteBuffer, Integer> vocabIndex = new HashMap<>();
        HashMap<Integer, String> vocabs = new HashMap<>();

        int index = 0;
        while ((line = brModel.readLine()) != null){
            String[] fields = line.split(" ");
            if (fields.length < vec_size) continue;

            vocabIndex.put(ByteBuffer.wrap(fields[0].getBytes()), index);
            vocabs.put(index, fields[0]);
            for (int i = 0; i < vec_size; i++){
                vectors[index][i] = Double.parseDouble(fields[1 + i]);
            }
            index++;
        }

        ArrayList<ByteBuffer> predicts = new ArrayList<>();
        int[] lookbacks = new int[window];
        int look_size = 0;

        //calculate all distances first;
        System.out.println("Vocab Size: " + vocab_size);
        int[][] topKId = new int[vocab_size][number];
        float[][] topKDist = new float[vocab_size][number];

        String topKFileId = outputFile + (".top" + number + ".Id");
        String topKFileDist = outputFile + (".top" + number + ".Dist");

        File topFileId = new File(topKFileId);
        File topFileDist = new File(topKFileDist);

        if (topFileDist.exists() && topFileId.exists()){

            System.out.println("Load From disk");
            BufferedReader br1 = new BufferedReader(new FileReader(topKFileId));
            BufferedReader br2 = new BufferedReader(new FileReader(topKFileDist));
            int i = 0;

            while ((line = br1.readLine()) != null){
                String[] fields = line.split(" ");
                for (int k = 0; k < number; k++){
                    topKId[i][k] = Integer.parseInt(fields[k]);
                }
                i++;
            }

            i = 0;
            while ((line = br2.readLine()) != null){
                String[] fields = line.split(" ");
                for (int k = 0; k < number; k++){
                    topKDist[i][k] = Float.parseFloat(fields[k]);
                }
                i++;
            }

        } else {
            double[][] normalize = new double[vocab_size][vec_size];
            for (int a = 0; a < vocab_size; a++) {
                double len = 0;
                for (int b = 0; b < vec_size; b++) len += vectors[a][b] * vectors[a][b];
                len = Math.sqrt(len);
                for (int b = 0; b < vec_size; b++) normalize[a][b] = vectors[a][b] / len;
            }

            for (int i = 0; i < vocab_size; i++) {
                for (int j = 0; j < vocab_size; j++) {
                    if (i != j) {
                        float dist = 0;
                        for (int b = 0; b < vec_size; b++) dist += normalize[i][b] * normalize[j][b];
                        for (int b = 0; b < number; b++) {
                            if (dist > topKDist[i][b]) {
                                for (int c = number - 1; c > b; c--) {
                                    topKDist[i][c] = topKDist[i][c - 1];
                                    topKId[i][c] = topKId[i][c - 1];
                                }
                                topKDist[i][b] = dist;
                                topKId[i][b] = j;
                                break;
                            }
                        }
                    }
                }
            }


            BufferedWriter bw1 = new BufferedWriter(new FileWriter(topKFileId));
            BufferedWriter bw2 = new BufferedWriter(new FileWriter(topKFileDist));

            for (int i = 0; i < vocab_size; i++) {
                String output = "";
                for (int j = 0; j < number; j++) output += (topKId[i][j] + " ");
                output += "\n";
                bw1.write(output);
            }
            bw1.close();

            for (int i = 0; i < vocab_size; i++) {
                String output = "";
                for (int j = 0; j < number; j++) output += (topKDist[i][j] + " ");
                output += "\n";
                bw2.write(output);
            }
            bw2.close();
        }

        System.out.println("Finish Calculate all distances");

        for (int p = 1; p <= 10; p++) {
            float k  = (float) 1 / (float) 10 * (float) p;
            int hit = 0;
            int total = 0;
            look_size = 0;

            BufferedReader br = new BufferedReader(new FileReader(verifyFile));
            while ((line = br.readLine()) != null) {
                String[] sentence = line.split(" ");
                int processed = 0;

                for (String word : sentence) {
                    processed++;

                    Integer rtn = vocabIndex.get(ByteBuffer.wrap(word.getBytes()));

                    if (rtn == null)
                        continue;
                    if (rtn == 0)
                        continue;

                    index = rtn;

                    if (predicts.contains(ByteBuffer.wrap(word.getBytes()))) {
                        hit++;
                    }
                    if (!predicts.isEmpty())
                        total++;

                    if (look_size < window)
                        lookbacks[look_size++] = index;
                    else {
                        for (int i = 0; i < window - 1; i++)
                            lookbacks[i] = lookbacks[i + 1];
                        lookbacks[window - 1] = index;
                    }

                    TreeSet<PredictPoss> sorted = new TreeSet<>();
                    predicts.clear();
                    float alpha = 1;

                    int trueNumber = 5;
                    for (int i = window - 1; i >= 0; i--) {
                        int context = lookbacks[i];
                        for (int a = 0; a < trueNumber; a++) {
                            String pid = vocabs.get(topKId[context][a]);
                            float poss = topKDist[context][a];
                            if (pid.equalsIgnoreCase("</s>"))
                                continue;
                            sorted.add(new PredictPoss(pid, poss * alpha));
                        }
                        alpha *= k;
                    }
                    if (processed == sentence.length)
                        continue;

                    for (int a = 0; a < trueNumber; a++) {
                        predicts.add(ByteBuffer.wrap(sorted.pollFirst().nodeId.getBytes()));
                    }
                }

            }
            br.close();
            System.out.println("Number: " + number + " p: " + p + " k " + k +  " hit: " + hit + " total: " + total +
                    " ratio:" +
                    " " +
                    (double) hit /
                    (double) total * 100 + " %");
        }

    }

    private static class PredictPoss implements Comparable<PredictPoss>{
        String nodeId;
        float poss;
        public PredictPoss(String id, float p){
            this.nodeId = id;
            this.poss = p;
        }
        //larger one should go at the beginning of list
        @Override
        public int compareTo(PredictPoss o) {
            if (this.poss - o.poss > 0)
                return -1;
            else if (this.poss - o.poss < 0)
                return 1;
            return 0;
        }
    }
}
