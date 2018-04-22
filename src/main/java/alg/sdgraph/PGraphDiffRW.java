package alg.sdgraph;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

/**
 * Created by daidong on 2/21/16.
 */
public class PGraphDiffRW {

	String trainFile;
	String verifyFile;
	String outputFile;
	final int cutting_window = 10 * 1000 * 1000 * 10; // 1s

	public PGraphDiffRW(String path, String verify, String out){
		this.trainFile = path;
		this.verifyFile = verify;
		this.outputFile = out;
	}

	ArrayList<String> lines = new ArrayList<>();
	ArrayList<String> timelines = new ArrayList<>();
	ArrayList<String> ops = new ArrayList<>();
	long maxBlockId = 0L;

	public void loadTrainFile() throws IOException, InterruptedException {

		BufferedReader br = new BufferedReader(new FileReader(this.trainFile));
		String line;
		long previous = 0;
		StringBuilder sentenceSB = new StringBuilder();
		StringBuilder timelineSB = new StringBuilder();
		StringBuilder opsSB = new StringBuilder();

		while ((line = br.readLine()) != null) {
			String[] fields = line.split(",");
			long ts = Long.parseLong(fields[0]);
			String type = fields[3];
			long offset = Long.parseLong(fields[4]);
			int size = Integer.parseInt(fields[5]);

			if (ts - previous > cutting_window && previous != 0) {
				lines.add(sentenceSB.toString());
				timelines.add(timelineSB.toString());
				ops.add(opsSB.toString());

				sentenceSB = new StringBuilder();
				timelineSB = new StringBuilder();
				opsSB = new StringBuilder();
			}

			long diskId = offset / 4096;
			long op = 0;
			if (type.equalsIgnoreCase("READ")) op = 0;
			else op = 1;
			diskId = diskId * 10 + op;

			/*
			 * Do not extract large block accesses.
			 * Record sequential access for prediction.
			 */
			int blocks = (int) Math.ceil((float) size / (float) 4096);

			sentenceSB.append(diskId + " ");
			timelineSB.append(ts + " ");
			opsSB.append(type + " ");

			previous = ts;
		}

		br.close();
	}

	public class Edge implements Comparable<Edge> {
		long src;
		long dst;
		int weight;
		double prob;

		public Edge(long s, long d){
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
			return (int) (src + dst);
		}

		@Override
		public boolean equals(Object obj) {
			if (obj instanceof Edge){
				Edge that = (Edge) obj;
				if (this.src == that.src &&
						this.dst == that.dst){
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
	private long[] window = new long[lookahead];
	private int window_size = 0;
	private ArrayList<Long> vertices = new ArrayList<>();
	private HashMap<Long, ArrayList<Edge>> edges = new HashMap<>();

	public void addToWindow(long blockId){
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

	public void addToVertices(long blockId){
		if (this.vertices.contains(blockId))
			return;
		else
			this.vertices.add(blockId);
	}

	public void addToEdges(long src, long dst){
		if (this.edges.containsKey(src)){
			ArrayList<Edge> lst = this.edges.get(src);
			boolean c = false;
			for (Edge e : lst){
				if (e.dst == dst) {
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
		for (int k = 0; k < lines.size(); k++){
			String line = lines.get(k);

			String[] ioaccess = line.split("\\s");
			emptyWindow();

			for (int i = 0; i < ioaccess.length; i++){
				String io = ioaccess[i];
				if (io.equalsIgnoreCase(""))
					continue;
				long blockId = Long.parseLong(io);
				//blockId = blockId;
				addToWindow(blockId);
				addToVertices(blockId);
				for (int j = 0; j < window_size - 1; j++){
					if (blockId != window[j])
						addToEdges(window[j], blockId);
				}
			}
			if ((k + 1) % 10000 == 0)
				System.out.println("training " + ((double) k / (double) lines.size() * 100) + "%");
		}

		for (long blockId : this.edges.keySet()){
			ArrayList<Edge> neighbors = this.edges.get(blockId);
			Collections.sort(neighbors);
		}

		BufferedWriter bw = new BufferedWriter(new FileWriter(this.outputFile));
		for (long bid : edges.keySet()){
			StringBuilder sb = new StringBuilder();
			sb.append(String.valueOf(bid) + ",");
			for (Edge e : edges.get(bid)){
				sb.append(String.valueOf(e.dst + " " + e.weight + ","));
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
		ArrayList<Long> predicts = new ArrayList<>();

		while ((line = br.readLine()) != null) {
			String[] fields = line.split(",");
			long ts = Long.parseLong(fields[0]);
			String type = fields[3];
			long offset = Long.parseLong(fields[4]);
			int size = Integer.parseInt(fields[5]);
			int response = Integer.parseInt(fields[6]);

			long diskId = offset / 4096;

			long op = 0;
			if (type.equalsIgnoreCase("READ")) op = 0;
			else op = 1;
			diskId = diskId * 10 + op;

			int blocks = (int) Math.ceil((float) size / (float) 4096);
			total++;

			if (predicts.contains(diskId)) {
				hit++;
			}

			predicts.clear();
			if (this.edges.containsKey(diskId)){
				ArrayList<Edge> neighbors = this.edges.get(diskId);
				for (Edge e : neighbors){
					if (predicts.size() < number)
						predicts.add(e.dst);
				}
			}
		}
		System.out.println("Number: " + number + " hit: " + hit + " total: " + total + " ratio: " + (double) hit /
				(double) total * 100 + " %");
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		PGraphDiffRW pg = new PGraphDiffRW(args[0], args[1], args[2]);
		pg.loadTrainFile();
		System.out.println("Load File Finished");
		pg.train();

	}

}
