package trace.snia.ms;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

/**
 * Created by daidong on 2/21/16.
 */
public class PGraphEvaluateFrequent {

	String trainFile;
	String verifyFile;
	String outputFile;
	final int cutting_window = 10 * 1000 * 1000 * 10; // 1s
	final int min_count = 5; //min_count
	HashMap<ByteBuffer, Integer> actives = new HashMap<>();

	public PGraphEvaluateFrequent(String path, String verify, String out){
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

	public class Edge implements Comparable<Edge> {
		ByteBuffer src;
		ByteBuffer dst;
		int weight;
		double prob;

		public Edge(ByteBuffer s, ByteBuffer d){
			this.src = s;
			this.dst = d;
			this.weight = 1;
			this.prob = 0.0;
		}
		public void incrWeight(){
			this.weight += 1;
		}
		public int getWeight(){
			return this.weight;
		}
		@Override
		public int hashCode() {
			return (int) (src.hashCode() + dst.hashCode());
		}

		@Override
		public boolean equals(Object obj) {
			if (obj instanceof Edge){
				Edge that = (Edge) obj;
				if (this.src.equals(that.src) &&
						this.dst.equals(that.dst)){
					return true;
				}
			}
			return false;
		}

		@Override
		public int compareTo(Edge o) {
			return (0 - (this.weight - o.weight));
		}
	}

	private final int lookahead = 5;
	private ByteBuffer[] window = new ByteBuffer[lookahead];
	private int window_size = 0;
	private ArrayList<ByteBuffer> vertices = new ArrayList<>();
	private HashMap<ByteBuffer, ArrayList<Edge>> edges = new HashMap<>();

	public void addToWindow(ByteBuffer blockId){
		if (window_size < lookahead){
			window[window_size++] = blockId;
		} else {
			for (int i = 0; i < (lookahead - 1); i++) {
				window[i] = window[i + 1];
			}
			window[lookahead - 1] = blockId;
		}
	}
	public void emptyWindow(){
		window_size = 0;
	}

	public void addToVertices(ByteBuffer blockId){
		if (this.vertices.contains(blockId))
			return;
		else
			this.vertices.add(blockId);
	}

	public void addToEdges(ByteBuffer src, ByteBuffer dst){

		if (this.edges.containsKey(src)){
			ArrayList<Edge> lst = this.edges.get(src);
			boolean c = false;
			for (Edge e : lst){
				if (e.dst.equals(dst)) {
					e.incrWeight();
					c = true;
					break;
				}
			}
			if (!c){
				Edge e = new Edge(src, dst);
				lst.add(e);
			}
		} else { //first occurance of source vertex
			ArrayList<Edge> lst = new ArrayList<>();
			Edge e = new Edge(src, dst);
			lst.add(e);
			this.edges.put(src, lst);
		}
	}

	public void train() throws IOException {
		System.out.println("Training Start: " + System.currentTimeMillis());
		int trained = 0;
		int total = 0;
		for (int k = 0; k < lines.size(); k++) {
			StringBuilder newLineSB = new StringBuilder();
			String line = lines.get(k);
			for (String w : line.split("\\s")){
				ByteBuffer bw = ByteBuffer.wrap(w.getBytes());
				if (actives.containsKey(bw) && actives.get(bw) >= min_count){
					total++;
					newLineSB.append(w + " ");
				}
			}
			if (newLineSB.length() > 0){
				newLineSB.append("\n");
				activeLines.add(newLineSB.toString());
			}
		}

		System.out.println("active access stream size: " + total);

		for (int k = 0; k < activeLines.size(); k++){
		//for (int k = 0; k < 30; k++){
			String line = activeLines.get(k);
			String[] ioaccess = line.split("\\s");
			emptyWindow();

			for (int i = 0; i < ioaccess.length; i++){
				String io = ioaccess[i];
				trained++;
				if (io.equalsIgnoreCase(""))
					continue;
				ByteBuffer blockId = ByteBuffer.wrap(io.getBytes());
				addToWindow(blockId);
				addToVertices(blockId);
				for (int j = 0; j < window_size - 1; j++){
					if (!blockId.equals(window[j]))
						addToEdges(window[j], blockId);
				}
				if ((trained) % 1000 == 0)
					System.out.println("training " + ((double) trained / (double) total * 100) + "%");
			}
		}

		System.out.println("Training Finish: " + System.currentTimeMillis());

		for (ByteBuffer blockId : this.edges.keySet()){
			ArrayList<Edge> neighbors = this.edges.get(blockId);
			Collections.sort(neighbors);
		}

		BufferedWriter bw = new BufferedWriter(new FileWriter(this.outputFile));
		for (ByteBuffer bid : edges.keySet()){
			StringBuilder sb = new StringBuilder();
			sb.append(ByteBufferToString(bid) + " ");
			for (Edge e : edges.get(bid)){
				sb.append(ByteBufferToString(e.dst) + " " + e.weight + " ");
			}
			sb.append("\n");
			bw.write(sb.toString());
		}
		bw.close();


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
				if (this.edges.containsKey(bw)) {
					ArrayList<Edge> neighbors = this.edges.get(bw);
					for (Edge e : neighbors) {
						if (predicts.size() < number)
							predicts.add(e.dst);
					}
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
				if (this.edges.containsKey(bw)) {
					ArrayList<Edge> neighbors = this.edges.get(bw);
					for (Edge e : neighbors) {
						if (predicts.size() < number)
							predicts.add(e.dst);
					}
				}
			}
		}
		System.out.println("Number: " + number + " hit: " + hit + " total: " + total + " ratio: " + (double) hit /
				(double) total * 100 + " %");
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		PGraphEvaluateFrequent pg = new PGraphEvaluateFrequent(args[0], args[1], args[2]);
		pg.loadTrainFile();
		System.out.println("Load File Finished");
		pg.train();

	}

}
