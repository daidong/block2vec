package trace.snia.ms;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

/**
 * Created by daidong on 2/21/16.
 */
public class SPEvaluateFrequent {

	String trainFile;
	String verifyFile;
	String outputFile;
	final int cutting_window = 10 * 1000 * 1000 * 10; // 1s
	final int min_count = 5; //min_count
	HashMap<ByteBuffer, Integer> actives = new HashMap<>();

	public SPEvaluateFrequent(String path, String verify, String out){
		this.trainFile = path;
		this.verifyFile = verify;
		this.outputFile = out;
	}

	public static String ByteBufferToString(ByteBuffer myByteBuffer){
		if (myByteBuffer.hasArray()) {
			return new String(myByteBuffer.array(),
					myByteBuffer.arrayOffset() + myByteBuffer.position(),
					myByteBuffer.remaining());
		} else {
			final byte[] b = new byte[myByteBuffer.remaining()];
			myByteBuffer.duplicate().get(b);
			return new String(b);
		}
	}

	ArrayList<String> lines = new ArrayList<>();
	ArrayList<String> activeLines = new ArrayList<>();
	long maxBlockId = 0L;

	public void loadTrainFile() throws IOException, InterruptedException {
		BufferedReader br = new BufferedReader(new FileReader(this.trainFile));
		String line;

		while ((line = br.readLine()) != null) {
			lines.add(line);
			for (String word : line.split("\\s")){
				ByteBuffer k = ByteBuffer.wrap(word.getBytes());
				if (!actives.containsKey(k))
					actives.put(k, 1);
				else
					actives.put(k, actives.get(k) + 1);
			}
		}
		br.close();
	}

	public void statistic() {
		int[] statistic = new int[50];
		for (ByteBuffer k : actives.keySet()){
			int t = actives.get(k);
			for (int i = Math.min(49, t); i > 0; i--)
				statistic[i] = statistic[i] + 1;
		}
		for (int i = 0; i < 50; i++)
			System.out.println(statistic[i]);
	}

	public void train() throws IOException {

		System.out.println("Accuracy 1");
		for (int number = 30; number > 0; number -= 5)
			accuracy1(number);

		System.out.println("Accuracy 2");
		for (int number = 30; number > 0; number -= 5)
			accuracy2(number);

	}

	public void accuracy1(int number) throws IOException {

		int hit = 0;
		int total = 0;

		BufferedReader br = new BufferedReader(new FileReader(this.verifyFile));
		String line;
		ArrayList<ByteBuffer> predicts = new ArrayList<>();

		while ((line = br.readLine()) != null) {
			String[] sentence = line.split(" ");
			int processed = 0;

			for (String word : sentence) {
				processed++;
				ByteBuffer bw = ByteBuffer.wrap(word.getBytes());
				if (predicts.contains(ByteBuffer.wrap(word.getBytes()))) {
					hit++;
				}
				if (!predicts.isEmpty())
					total++;

				predicts.clear();

				int bId = Integer.parseInt(word.split(":")[0]);
				for (int k = 0; k < number; k++) {
					String bidRead = (bId + k) + ":" + "R";
					String bidWrite = (bId + k) + ":" + "W";
					predicts.add(ByteBuffer.wrap(bidRead.getBytes()));
					predicts.add(ByteBuffer.wrap(bidWrite.getBytes()));

				}
			}
		}
		System.out.println("Number: " + number + " hit: " + hit + " total: " + total + " ratio: " + (double) hit /
				(double) total * 100 + " %");
	}

	public void accuracy2(int number) throws IOException {

		int hit = 0;
		int total = 0;

		BufferedReader br = new BufferedReader(new FileReader(this.verifyFile));
		String line;
		ArrayList<ByteBuffer> predicts = new ArrayList<>();

		while ((line = br.readLine()) != null) {
			String[] sentence = line.split(" ");
			int processed = 0;

			for (String word : sentence) {
				processed++;
				ByteBuffer bw = ByteBuffer.wrap(word.getBytes());
				if (!actives.containsKey(bw) || actives.get(bw) < min_count)
					continue;
				if (predicts.contains(ByteBuffer.wrap(word.getBytes()))) {
					hit++;
				}
				if (!predicts.isEmpty())
					total++;

				predicts.clear();
				int bId = Integer.parseInt(word.split(":")[0]);
				for (int k = 0; k < number; k++) {
					String bidRead = (bId + k) + ":" + "R";
					String bidWrite = (bId + k) + ":" + "W";
					predicts.add(ByteBuffer.wrap(bidRead.getBytes()));
					predicts.add(ByteBuffer.wrap(bidWrite.getBytes()));

				}
			}
		}
		System.out.println("Number: " + number + " hit: " + hit + " total: " + total + " ratio: " + (double) hit /
				(double) total * 100 + " %");
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		SPEvaluateFrequent pg = new SPEvaluateFrequent(args[0], args[1], args[2]);
		pg.loadTrainFile();
		System.out.println("Load File Finished");
		pg.statistic();
		//pg.train();

	}

}
