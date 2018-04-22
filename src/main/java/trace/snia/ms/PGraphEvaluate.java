package trace.snia.ms;

import com.sun.javafx.image.ByteToBytePixelConverter;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

/**
 * Created by daidong on 2/21/16.
 */
public class PGraphEvaluate {

	String trainFile;
	String verifyFile;
	String outputFile;
	final int cutting_window = 10 * 1000 * 1000 * 10; // 1s

	public PGraphEvaluate(String path, String verify, String out){
		this.trainFile = path;
		this.verifyFile = verify;
		this.outputFile = out;
	}

	ArrayList<String> lines = new ArrayList<>();
	long maxBlockId = 0L;

	public void loadTrainFile() throws IOException, InterruptedException {

		BufferedReader br = new BufferedReader(new FileReader(this.trainFile));
		String line;

		while ((line = br.readLine()) != null) {
			lines.add(line);
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
		int trained = 0;
		int total = 0;
		for (int k = 0; k < lines.size(); k++) {
			String line = lines.get(k);
			total += line.split("\\s").length;
		}
		for (int k = 0; k < lines.size(); k++){
			String line = lines.get(k);
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

		for (ByteBuffer blockId : this.edges.keySet()){
			ArrayList<Edge> neighbors = this.edges.get(blockId);
			Collections.sort(neighbors);
		}

		BufferedWriter bw = new BufferedWriter(new FileWriter(this.outputFile));
		for (ByteBuffer bid : edges.keySet()){
			StringBuilder sb = new StringBuilder();
			sb.append(bid.toString() + ",");
			for (Edge e : edges.get(bid)){
				sb.append(String.valueOf(e.dst) + " " + e.weight + ",");
			}
			sb.append("\n");
			bw.write(sb.toString());
		}
		bw.close();

		for (int number = 30; number > 0; number -= 5)
			accuracy(number);
	}

	public void accuracy(int number) throws IOException {

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

				if (predicts.contains(ByteBuffer.wrap(word.getBytes()))) {
					hit++;
				}
				if (!predicts.isEmpty())
					total++;

				predicts.clear();
				if (this.edges.containsKey(word.getBytes())) {
					ArrayList<Edge> neighbors = this.edges.get(word.getBytes());
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
		PGraphEvaluate pg = new PGraphEvaluate(args[0], args[1], args[2]);
		pg.loadTrainFile();
		System.out.println("Load File Finished");
		pg.train();

	}

}
