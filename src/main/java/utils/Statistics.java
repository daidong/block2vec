package utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by daidong on 2/21/16.
 */
public class Statistics {

	String trainFile;

	public class BlockAccess {
		long ts;
		long blockId;
		int size;
		int type = 0;

		public BlockAccess(long t, long b, int s, int type){
			this.ts = t;
			this.blockId = b;
			this.size = s;
			this.type = type;
		}
	}

	ArrayList<BlockAccess> blks = new ArrayList<>();
	public Statistics(String path){
		this.trainFile = path;
	}

	public void LoadFile() throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(this.trainFile));
		String line;

		while ((line = br.readLine()) != null) {
			String[] fields = line.split(",");
			long ts = Long.parseLong(fields[0]);
			String type = fields[3];
			long offset = Long.parseLong(fields[4]);
			int size = Integer.parseInt(fields[5]);
			int response = Integer.parseInt(fields[6]);

			long diskId = offset / 4096;

			int blocks = (int) Math.ceil((float) size / (float) 4096);

			int t = 0;
			if (type.equalsIgnoreCase("READ")) t = 0;
			else t = 1;

			blks.add(new BlockAccess(ts, diskId, blocks, t));
		}

		br.close();
	}
	public int frequentAccess(int f){
		int num = 0;
		HashMap<Long, Integer> blockFrequency = new HashMap<>();
		for (BlockAccess b : blks){
			if (!blockFrequency.containsKey(b.blockId))
				blockFrequency.put(b.blockId, 0);
			int t = blockFrequency.get(b.blockId);
			blockFrequency.put(b.blockId, t+1);
		}
		for (long id : blockFrequency.keySet())
			if (blockFrequency.get(id) > f)
				num++;
		return num;
	}

	public static void main(String[] args) throws IOException {
		Statistics s = new Statistics(args[0]);
		s.LoadFile();
		for (int f = 1; f < 50; f++)
			System.out.println("f: " + f + " number: " + s.frequentAccess(f));
	}
}
