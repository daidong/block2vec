package alg.sequence;

import utils.JenkinsHash;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.TreeSet;

/**
 * Created by daidong on 2/18/16.
 */
public class PureSequentialPredict {

    String trainFile;
    String outputFile;
    //MSR Time Trace use "Windows filetime" to record the timestamp.
    //A file time is a 64-bit value that represents the number of 100-nanosecond intervals
    // that have elapsed since 12:00 A.M. January 1, 1601 Coordinated Universal Time (UTC).
    //https://msdn.microsoft.com/en-us/library/windows/desktop/ms724290(v=vs.85).aspx
    final int cutting_window = 10 * 1000 * 1000 * 1; // 1s
    final int time_sensitive_window = 10 * 1000 * 10; // 10ms

    public PureSequentialPredict(String input, String output){
        this.trainFile = input;
        this.outputFile = output;
    }

    public void accuracy(int number) throws IOException {

        int hit = 0;
        int total = 0;

        BufferedReader br = new BufferedReader(new FileReader(this.trainFile));
        String line;

        TreeSet<Long> predict = new TreeSet<>();

        while ((line = br.readLine()) != null) {
            String[] fields = line.split(",");
            long ts = Long.parseLong(fields[0]);
            String type = fields[3];
            long offset = Long.parseLong(fields[4]);
            int size = Integer.parseInt(fields[5]);
            int response = Integer.parseInt(fields[6]);

            if (!type.equalsIgnoreCase("READ"))
                continue;

            long diskId = offset / 4096;
            int blocks = (int) Math.ceil((float) size / (float) 4096);
            //total += blocks;
            total += 1;

            if (predict.contains(diskId))
                hit++;

            //if (blocks > 1)
            //    hit += (blocks - 1);

            //diskId += blocks;
            predict.clear();

            for (int i = 0; i < number; i++)
                predict.add(diskId + i);

        }

        System.out.println("Next: " + number + " hit: " + hit + " total: " + total + " ratio: " + (double)hit/
                (double) total * 100 + " %");
        br.close();
    }

    public void train() throws IOException {
        System.out.println("Starting sequential training using file " + this.trainFile);

        for (int number = 200; number > 0; number -= 20)
            accuracy(number);
        accuracy(5);

        System.out.println("Finish training, output: " + this.outputFile);
    }

    public static void main(String[] args) throws IOException {
        PureSequentialPredict instance = new PureSequentialPredict(args[0], args[1]);
        instance.train();
    }
}
