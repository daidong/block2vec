package trace.snia.ms;

import java.io.*;

/**
 * Created by daidong on 2/22/16.
 */
public class GenAllTrainingEvalFiles {

	private static String getFileExtension(File file) {
		String name = file.getName();
		try {
			return name.substring(name.lastIndexOf(".") + 1);
		} catch (Exception e) {
			return "";
		}
	}

	public static void main(String[] args) throws IOException {
		String dir = args[0];
		double percentage = Double.parseDouble(args[1]);

		File[] files = new File(dir).listFiles();
		for (File file : files) {
			if (!file.isFile()) continue;
			if (!getFileExtension(file).equalsIgnoreCase("data")) continue;
			System.out.println("Start Processing " + file.getName());
			String trainingFile = file.getAbsolutePath() + ".train";
			String verifyFile = file.getAbsolutePath() + ".verify";

			BufferedReader br = new BufferedReader(new FileReader(file));
			BufferedWriter bw1 = new BufferedWriter(new FileWriter(trainingFile));
			BufferedWriter bw2 = new BufferedWriter(new FileWriter(verifyFile));

			String line;
			int words = 0;
			while ((line = br.readLine()) != null) {
				String[] fields = line.split(" ");
				words += fields.length;
			}
			int trainingWords = (int) (words * percentage);
			words = 0;
			br.close();
			br = new BufferedReader(new FileReader(file));
			while ((line = br.readLine()) != null) {
				String[] fields = line.split(" ");
				words += fields.length;

				if (words > trainingWords)
					break;

				bw1.write(line + "\n");
			}
			bw1.close();

			while ((line = br.readLine()) != null) {
				bw2.write(line + "\n");
			}

			br.close();
			bw2.close();
		}
	}

}
