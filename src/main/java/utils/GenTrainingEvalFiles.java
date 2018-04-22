package utils;

import java.io.*;

/**
 * Created by daidong on 2/22/16.
 */
public class GenTrainingEvalFiles {
	public static void main(String[] args) throws IOException {
		int trainingDays = 6;
		//long days = trainingDays * 24 * 60 * 60 * 1000 * 1000 * 10L;
		long days = 5184000000000L;
		System.out.println("days: " + days);

		BufferedReader br = new BufferedReader(new FileReader(args[0]));
		BufferedWriter bw1 = new BufferedWriter(new FileWriter(args[1]));
		BufferedWriter bw2 = new BufferedWriter(new FileWriter(args[2]));

		String line;
		long start = 0L;
		while ((line = br.readLine()) != null) {
			String[] fields = line.split(",");
			long ts = Long.parseLong(fields[0]);
			String type = fields[3];
			long offset = Long.parseLong(fields[4]);
			int size = Integer.parseInt(fields[5]);

			if (start == 0)
				start = ts;

			long timeDiff = ts - start;
			if (timeDiff < days){
				bw1.write(line+"\n");
			} else {
				bw2.write(line+"\n");
			}

		}
		br.close();
		bw1.close();
		bw2.close();
	}

}
