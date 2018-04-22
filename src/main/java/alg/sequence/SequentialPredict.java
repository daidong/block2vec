package alg.sequence;

import utils.JenkinsHash;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

/**
 * Created by daidong on 2/18/16.
 */
public class SequentialPredict {

    private class Word implements Comparable<Word> {
        public long cn;
        public int[] point;
        public String word;
        public char[] code;
        public char codeLen;

        public Word(String w) {
            this.word = w;
            this.cn = 1;
            this.point = new int[max_code_length];
            this.code = new char[max_code_length];
        }

        // reversed natural order, </s> keeps at the beginning of the vocab
        public int compareTo(Word o) {
            if (this.word.equals("</s>") && !o.word.equals("</s>"))
                return -1;
            if (!this.word.equals("</s>") && o.word.equals("</s>"))
                return 1;
            return (int) (o.cn - this.cn);
        }

        @Override
        public String toString() {
            return "Word{" +
                    "cn=" + cn +
                    ", word='" + word + '\'' +
                    ", code=" + Arrays.toString(code) +
                    '}';
        }
    }

    String trainFile;
    String outputFile;
    //MSR Time Trace use "Windows filetime" to record the timestamp.
    //A file time is a 64-bit value that represents the number of 100-nanosecond intervals
    // that have elapsed since 12:00 A.M. January 1, 1601 Coordinated Universal Time (UTC).
    //https://msdn.microsoft.com/en-us/library/windows/desktop/ms724290(v=vs.85).aspx
    final int cutting_window = 10 * 1000 * 1000 * 1; // 1s
    final int time_sensitive_window = 10 * 1000 * 10; // 10ms
    float[] syn0;
    int vocab_size = 0;
    int vocab_hash_size = 30000000;
    int max_code_length = 40;
    int min_reduce = 1;
    int train_words = 0;
    int min_count = 5;
    int layer1_size = 1;


    ArrayList<Word> vocab = new ArrayList<>();
    int[] vocab_hash = new int[vocab_hash_size];


    public SequentialPredict(String input, String output){
        this.trainFile = input;
        this.outputFile = output;
    }

    ArrayList<String> lines = new ArrayList<>();
    ArrayList<String> timelines = new ArrayList<>();
    long maxBlockId = 0L;

    public void loadTrainFile() throws IOException {

        BufferedReader br = new BufferedReader(new FileReader(this.trainFile));
        String line;
        long previous = 0;
        StringBuilder sentenceSB = new StringBuilder();
        StringBuilder timelineSB = new StringBuilder();

        while ((line = br.readLine()) != null) {
            String[] fields = line.split(",");
            long ts = Long.parseLong(fields[0]);
            long offset = Long.parseLong(fields[4]);
            int size = Integer.parseInt(fields[5]);
            int response = Integer.parseInt(fields[6]);

            if (ts - previous > cutting_window && previous != 0){
                lines.add(sentenceSB.toString());
                timelines.add(timelineSB.toString());
                sentenceSB = new StringBuilder();
                timelineSB = new StringBuilder();
            }

            long diskId = offset / 4096;

            int blocks = (int) Math.ceil((float) size / (float) 4096);

            for (int i = 0; i < blocks; i++) {
                sentenceSB.append((diskId + i) + " ");
                timelineSB.append((ts + i) + " ");
            }
            if ((diskId + blocks) > maxBlockId)
                maxBlockId = (diskId + blocks);

            previous = ts;
        }

        br.close();
    }

    public int getWordHash(String word) {
        JenkinsHash jh = new JenkinsHash();
        return (Math.abs(jh.hash32(word.getBytes())) % vocab_hash_size);
    }

    public int searchVocab(String word) {
        int hash = getWordHash(word);
        while (true) {
            if (vocab_hash[hash] == -1) return -1;
            if (word.equals(vocab.get(vocab_hash[hash]).word))
                return vocab_hash[hash];
            hash = (hash + 1) % vocab_hash_size;
        }
    }

    public int addWordToVocab(String word) {
        Word w = new Word(word);
        vocab.add(w);
        vocab_size++;
        int hash = getWordHash(word);
        while (vocab_hash[hash] != -1)
            hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = vocab.size() - 1;
        return vocab.size() - 1;
    }

    public void sortVocab() {
        //Collections.sort(vocab);
        for (int i = 0; i < vocab_hash_size; i++)
            vocab_hash[i] = -1;
        train_words = 0;
        ArrayList<Word> movs = new ArrayList<>();
        for (int i = 0; i < vocab.size(); i++) {
            if ((vocab.get(i).cn < min_count) && i != 0) {
                vocab_size--;
                movs.add(vocab.get(i));
            }
        }
        vocab.removeAll(movs);
        for (int i = 0; i < vocab.size(); i++) {
            Word w = vocab.get(i);
            int hash = getWordHash(w.word);
            while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
            vocab_hash[hash] = i;
            train_words += w.cn;
        }
    }

    public void reduceVocab() {
        System.out.println("Reduce Vocab Called");
        ArrayList<Word> movs = new ArrayList<>();
        for (Word w : vocab) {
            if (w.cn <= min_reduce)
                movs.add(w);
        }
        vocab.removeAll(movs);
        for (int i = 0; i < vocab_hash_size; i++)
            vocab_hash[i] = -1;
        for (int i = 0; i < vocab.size(); i++) {
            Word w = vocab.get(i);
            int hash = getWordHash(w.word);
            while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
            vocab_hash[hash] = i;
        }
        min_reduce++;
    }

    public void learnVocabFromTrainFile() throws IOException {
        System.out.println("Pre-Processing file: " + this.trainFile);

        for (int i = 0; i < vocab_hash_size; i++) vocab_hash[i] = -1;

        addWordToVocab("</s>");

        for (String line : lines) {
            String[] sentence = line.split("\\s");

            for (String word : sentence) {
                train_words++;
                int i = searchVocab(word);
                if (i == -1) {
                    int a = addWordToVocab(word);
                    vocab.get(a).cn = 1;
                } else
                    vocab.get(i).cn++;

                if (vocab.size() > vocab_hash_size * 0.7)
                    reduceVocab();
            }

            train_words++;
            String word = "</s>";
            int i = searchVocab(word);
            vocab.get(i).cn++;

        }
        sortVocab();
    }

    public void trainModel(){
        syn0 = new float[vocab.size() * layer1_size];
        for (int c = 0; c < vocab_size; c++) {
            long blockId = 0;
            try {
                blockId = Long.parseLong(vocab.get(c).word);
            } catch (NumberFormatException e){
                blockId = 0;
            }
            syn0[c * layer1_size] = (float) blockId / (float) (maxBlockId + 1);
        }
    }

    public void kmeans(int classes) throws IOException {
        System.out.println("Run KMeans on " + classes + " groups");
        int clcn = classes;
        int iter = 10, closeid;
        int centcn[] = new int[classes];
        ArrayList<Integer>[] clusters = new ArrayList[classes];
        for (int i = 0; i < classes; i++) clusters[i] = new ArrayList<Integer>();

        int cl[] = new int[(int) vocab_size];
        float closev, x;
        float cent[] = new float[classes];

        for (int a = 0; a < vocab_size; a++) {
            cl[a] = (int) (syn0[a] * classes) ;
        }

        for (int a = 0; a < iter; a++) {
            System.out.println("KMeans Progress: " + ((float) a / (float) iter) * 100 + " %");

            for (int b = 0; b < clcn; b++)
                cent[b] = 0;

            for (int b = 0; b < clcn; b++)
                centcn[b] = 1;

            for (int c = 0; c < vocab_size; c++) {
                for (int d = 0; d < layer1_size; d++)
                    cent[cl[c]] += syn0[c * layer1_size + d];
                centcn[cl[c]]++;
            }
            for (int b = 0; b < clcn; b++) {
                cent[layer1_size * b] /= centcn[b];
            }
            for (int c = 0; c < vocab_size; c++) {
                closev = 10;
                closeid = 0;
                for (int d = 0; d < clcn; d++) {
                    x = Math.abs(cent[layer1_size * d] - syn0[c * layer1_size]);
                    if (x < closev) {
                        closev = x;
                        closeid = d;
                    }
                }
                cl[c] = closeid;
            }
        }
        System.out.println("KMeans Progress: " + "100%");

        for (int c = 0; c < vocab_size; c++)
            clusters[cl[c]].add(c);

        BufferedWriter bw = new BufferedWriter(new FileWriter(this.outputFile+".group"+classes));
        bw.write(vocab.size() + " " + layer1_size + "\n");
        for (int i = 0; i < classes; i++){
            String outs = "";
            outs += ("class: " + i + ":\t{");
            for (int b : clusters[i]){
                outs += (vocab.get(b).word+",");
            }
            outs += "}\n";
            bw.write(outs);
        }
        bw.close();

        System.out.println("Group are persisted into: " + this.outputFile + ".group"+classes);

        int hit = 0;
        int unhit = 0;
        int total = 0;

        for (String line : lines) {
            String[] ioaccess = line.split("\\s");
            for (int i = 0; i < ioaccess.length; i++){
                String io = ioaccess[i];
                int index = searchVocab(io);
                if (index == -1)
                    continue;
                if (index == 0)
                    continue;

                int g = cl[index];
                if ((i + 1) < ioaccess.length){
                    String next = ioaccess[i+1];
                    int nextInt = searchVocab(next);
                    if (clusters[g].contains(nextInt)) {
                        hit++;
                    }
                    total++;
                }
            }
        }
        System.out.println("Group: " + classes + " hit: " + hit + " total: " + total + " ratio: " + (double)hit/
                (double) total * 100 + " %");
    }

    public void train() throws IOException {
        System.out.println("Starting training using file " + this.trainFile);
        loadTrainFile();

        learnVocabFromTrainFile();

        trainModel();
        for (int group = 1000; group < 10000; group += 1000)
            kmeans(group);

        System.out.println("Finish training, output: " + this.outputFile);
    }

    public static void main(String[] args) throws IOException {
        SequentialPredict instance = new SequentialPredict(args[0], args[1]);
        instance.train();
    }
}
