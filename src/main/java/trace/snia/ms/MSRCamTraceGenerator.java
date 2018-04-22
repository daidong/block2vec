package trace.snia.ms;

import java.io.*;

public class MSRCamTraceGenerator {

	public static final int block_size = 4096;
	//MSR Time Trace use "Windows filetime" to record the timestamp.
	//A file time is a 64-bit value that represents the number of 100-nanosecond intervals
	// that have elapsed since 12:00 A.M. January 1, 1601 Coordinated Universal Time (UTC).
	//https://msdn.microsoft.com/en-us/library/windows/desktop/ms724290(v=vs.85).aspx
	final static int cutting_window = 10 * 1000 * 1000 * 10; // 10 s

	public static void main(String[] args) throws IOException {

		String outputfile = args[1];
		String outputTime = args[2];

		File[] files = new File[1];
		files[0] = new File(args[0]);

		BufferedWriter bw = new BufferedWriter(new FileWriter(outputfile));
		BufferedWriter bwt = new BufferedWriter(new FileWriter(outputTime));

		for (File file : files) {
			System.out.println("Processing file " + file.getAbsolutePath());
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line;
			long previous = 0;
			StringBuilder sentence = new StringBuilder();
			StringBuilder time = new StringBuilder();

			while ((line = br.readLine()) != null) {
				String[] fields = line.split(",");
				long ts = Long.parseLong(fields[0]);
				String type = fields[3];
				long offset = Long.parseLong(fields[4]);
				int size = Integer.parseInt(fields[5]);

				if (ts - previous > cutting_window && previous != 0) {
					if (sentence.length() > 0) {
						sentence.append("\n");
						bw.write(sentence.toString());

						time.append("\n");
						bwt.write(time.toString());

						sentence.delete(0, sentence.length());
						time.delete(0, time.length());

					}
				}

				long blockId = offset / block_size;
				String word = "";
				if (type.equalsIgnoreCase("READ")) word = String.format("%d:%s ", (blockId), "R");
				else word = String.format("%d:%s ", (blockId), "W");

				sentence.append(word);
				time.append(ts + " ");

				previous = ts;
			}
			br.close();
		}
		bw.close();
		bwt.close();
	}
}
