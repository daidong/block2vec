package alg.sdgraph;

import trace.snia.ms.MSTraceLoader;

import java.util.*;

/**
 * Created by daidong on 6/10/15.
 * Based on paper "C-Miner: Mining Block Correlations in Storage Systems" in FAST04
 * 1. specify a lookahead window size, i.e., 100
 * 2. When an item B is accessed, it adds node B into graph if it is not in the graph yet
 * 3. increment the weight of edge (Bi, B) for all Bi in current window; if edge does not exist, initialize one
 * 4. after entire access stream is processed, algorithm rescans the SD graph and only records those coorelations with
 * the weight larger than given threshold.
 */
public class SDGraph {

    public class Edge implements Comparator<Edge>{
        String src;
        String dst;
        int weight;

        public Edge(String s, String d){
            this.src = s;
            this.dst = d;
            this.weight = 1;
        }
        public void incrWeight(){
            this.weight += 1;
        }
        public int getWeight(){
            return this.weight;
        }
        @Override
        public int hashCode() {
            return src.length() + dst.length();
        }

        @Override
        public int compare(Edge o1, Edge o2) {
            return (0 - (o1.weight - o2.weight));
        }

        @Override
        public boolean equals(Object obj) {
            if (obj instanceof Edge){
                Edge that = (Edge) obj;
                if (this.src.equalsIgnoreCase(that.src) &&
                    this.dst.equalsIgnoreCase(that.dst)){
                    return true;
                }
            }
            return false;
        }
    }

    private int lookahead = 100;
    private int threshold = 10;

    private ArrayList<String> vertices;
    private HashMap<String, TreeSet<Edge>> edges;

    private String[] window = new String[lookahead];

    private int startIdx, endIdx;
    private int windowSize;
    String trainFile;
    String outputFile;


    public SDGraph(String input, String output){
        this.trainFile = input;
        this.outputFile = output;
    }

    public SDGraph(int lookahead, int threshold){
        this.lookahead = lookahead;
        this.threshold = threshold;
        this.window = new String[this.lookahead];
        this.windowSize = 0;
        this.startIdx = 0;
        this.endIdx = this.lookahead - 1;
        this.vertices = new ArrayList<>();
        this.edges = new HashMap<>();
    }

    public void addToWindow(String blockId){
        if (this.windowSize < this.lookahead){
            window[windowSize] = blockId;
            windowSize++;
        } else {
            window[startIdx] = blockId;
            startIdx = (startIdx + 1) % this.lookahead;
            endIdx = (endIdx + 1) % this.lookahead;
        }
    }

    public void addToVertices(String blockId){
        if (this.vertices.contains(blockId))
            return;
        else
            this.vertices.add(blockId);
    }

    public void addToEdges(String src, String dst){
        if (this.edges.containsKey(src)){
            TreeSet<Edge> lst = this.edges.get(src);
            Edge e = new Edge(src, dst);
            if (lst.contains(e)){
                e.incrWeight();
                lst.add(e);
            } else {
                lst.add(e);
            }
        } else { //first occurance of source vertex
            TreeSet<Edge> lst = new TreeSet<>();
            Edge e = new Edge(src, dst);
            lst.add(e);
            this.edges.put(src, lst);
        }
    }

    public void train(){
        String blockId;
        while ((blockId = MSTraceLoader.getNextBlock()) != null){
            addToWindow(blockId);
            addToVertices(blockId);
            for (int i = startIdx; i <= endIdx; i = (i++)%this.lookahead){
                addToEdges(window[i], blockId);
            }
        }

        for (String src : this.edges.keySet()){
            TreeSet<Edge> lst = this.edges.get(src);
            List<Edge> nlst = new ArrayList<>(lst);
            for (Edge e : nlst){
                if (e.getWeight() < this.threshold){
                    lst.remove(e);
                }
            }
        }
    }

    public List<String> predict(String blockId, int predict_window){
        TreeSet<Edge> es = this.edges.get(blockId);
        int i = 0;
        List<String> rtn = new ArrayList<>();

        for (Edge e : es){
            if (i >= predict_window)
                return rtn;
            rtn.add(e.dst);
            i++;
        }

        return rtn;
    }

    public static void main(String[] args){
        SDGraph instance = new SDGraph(args[0], args[1]);
        instance.train();
    }
}
