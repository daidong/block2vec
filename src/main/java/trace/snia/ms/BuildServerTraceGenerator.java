package trace.snia.ms;

import java.io.*;

/**
 * BuildServer
 */
public class BuildServerTraceGenerator {
    public static final int block_size = 4096;
    public static final long block_window = 10000; //10,000 ms = 10 s

    public static void main(String[] args) throws IOException, InterruptedException {
        String dir = args[0];

        String outputfile = args[1];
        String outputTime = args[2];

        File[] files = new File(dir).listFiles();
        BufferedWriter bw = new BufferedWriter(new FileWriter(outputfile));
        BufferedWriter bwt = new BufferedWriter(new FileWriter(outputTime));


        for (File file : files) {
            System.out.println("Start Processing " + file.getName());
            BufferedReader br = new BufferedReader(new FileReader(file));
            String line;
            boolean dataBegin = false;

            StringBuilder sentence = new StringBuilder();
            StringBuilder time = new StringBuilder();

            long sentence_ts = 0;
            while ((line = br.readLine()) != null) {
                if (line.startsWith("EndHeader"))
                    dataBegin = true;
                if (!dataBegin)
                    continue;

                String[] fields = line.split(",");
                if (fields[0].equalsIgnoreCase("               DiskRead")) {
                    long ts = Long.parseLong(fields[1].trim());
                    if (ts - sentence_ts > MSTraceBlocks.block_window) {
                        if (sentence.length() > 0) {
                            sentence.append("\n");
                            bw.write(sentence.toString());

                            time.append("\n");
                            bwt.write(time.toString());

                            sentence.delete(0, sentence.length());
                            time.delete(0, time.length());

                        }
                    }
                    sentence_ts = ts;
                    long byteOffset = Long.parseLong(fields[5].trim().substring(2), 16);
                    long ioSize = Long.parseLong(fields[6].trim().substring(2), 16);
                    int diskNum = Integer.parseInt(fields[8].trim()); //

                    int blockId = (int) (byteOffset / MSTraceBlocks.block_size);
                    int range = (int) (ioSize / MSTraceBlocks.block_size);
                    String word = String.format("%d:%s ", (blockId), "R");
                    sentence.append(word);
                    time.append(ts + " ");
                }


                if (fields[0].equalsIgnoreCase("              DiskWrite")) {
                    long ts = Long.parseLong(fields[1].trim());
                    if (ts - sentence_ts > MSTraceBlocks.block_window) {
                        if (sentence.length() > 0) {
                            sentence.append("\n");
                            bw.write(sentence.toString());

                            time.append("\n");
                            bwt.write(time.toString());

                            sentence.delete(0, sentence.length());
                            time.delete(0, time.length());

                        }
                        sentence_ts = ts;
                    }
                    long byteOffset = Long.parseLong(fields[5].trim().substring(2), 16);
                    long ioSize = Long.parseLong(fields[6].trim().substring(2), 16);
                    int diskNum = Integer.parseInt(fields[8].trim()); //

                    int blockId = (int) (byteOffset / MSTraceBlocks.block_size);
                    int range = (int) (ioSize / MSTraceBlocks.block_size);
                    String word = String.format("%d:%s ", (blockId), "W");
                    sentence.append(word);
                    time.append(ts + " ");

                }
            }
            br.close();
        }
        bw.close();
        bwt.close();
    }
}
