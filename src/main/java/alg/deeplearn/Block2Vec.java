package alg.deeplearn;

import utils.CPrint;
import utils.JenkinsHash;

import java.io.*;
import java.util.*;

/**
 * A Simplied Java Version of Block2Vec Library
 *
 * @TODO: directly process trace(), load once, train multiple times. include read/write? (maybe next step)
 */
public class Block2Vec {


	private class Word implements Comparable<Word> {
		public long cn;
		public int[] point;
		public String word;
		public char[] code;
		public char codeLen;

		public Word(String w) {
			this.word = w;
			this.cn = 1;
			this.point = new int[max_code_length];
			this.code = new char[max_code_length];
		}

		// reversed natural order, </s> keeps at the beginning of the vocab
		public int compareTo(Word o) {
			if (this.word.equals("</s>") && !o.word.equals("</s>"))
				return -1;
			if (!this.word.equals("</s>") && o.word.equals("</s>"))
				return 1;
			return (int) (o.cn - this.cn);
		}

		@Override
		public String toString() {
			return "Word{" +
					"cn=" + cn +
					", word='" + word + '\'' +
					", code=" + Arrays.toString(code) +
					'}';
		}
	}

	String trainFile;
	String outputFile;

	//MSR Time Trace use "Windows filetime" to record the timestamp.
	//A file time is a 64-bit value that represents the number of 100-nanosecond intervals
	// that have elapsed since 12:00 A.M. January 1, 1601 Coordinated Universal Time (UTC).
	//https://msdn.microsoft.com/en-us/library/windows/desktop/ms724290(v=vs.85).aspx
	final int cutting_window = 10 * 10 * 1000 * 1; // 10ms; 1s is not good
	final int time_sensitive_window = 10 * 1000 * 10; // 10ms

	int vocab_hash_size = 30000000;
	int table_size = 100000000;
	int max_code_length = 40;
	int max_sentence_length = 1000;
	int EXP_TABLE_SIZE = 1000;
	int MAX_EXP = 6;
	//layer1_size should be less than 50. The less the better, I think!
	//int layer1_size = 200;
	int layer1_size = 50;

	ArrayList<Word> vocab = new ArrayList<>();
	int[] vocab_hash = new int[vocab_hash_size];
	long vocab_size = 0;
	int[] table = new int[table_size];

	float alpha = 0.025f, starting_alpha, sample = 0;
	float[] syn0, sync1, syn1neg, expTable;
	int negative = 25;
	int iter = 5;
	int window = 30;
	int min_count = 1;
	int min_reduce = 1;
	boolean binary = true;
	boolean hs = true;

	int train_words = 0;
	int word_count_actual = 0;

	public Block2Vec(String in, String out) {
		this.trainFile = in;
		this.outputFile = out;
		expTable = new float[EXP_TABLE_SIZE + 1];
		for (int i = 0; i < EXP_TABLE_SIZE; i++) {
			expTable[i] = (float) Math.exp((i / (float) EXP_TABLE_SIZE * 2 - 1) * (float) MAX_EXP);
			expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
		}
	}

	public void initUnigramTable() {
		double train_words_pow = 0.0;
		double power = 0.75;
		for (int a = 0; a < vocab.size(); a++)
			train_words_pow += Math.pow(vocab.get(a).cn, power);
		int i = 0;
		double d1 = Math.pow(vocab.get(i).cn, power) / train_words_pow;
		for (int a = 0; a < table_size; a++) {
			table[a] = i;
			if ((double) a / (double) table_size > d1) {
				i++;
				d1 += Math.pow(vocab.get(i).cn, power) / train_words_pow;
			}
			if (i >= vocab.size())
				i = (vocab.size() - 1);
		}
	}

	public int getWordHash(String word) {
		JenkinsHash jh = new JenkinsHash();
		return (Math.abs(jh.hash32(word.getBytes())) % vocab_hash_size);
	}

	public int searchVocab(String word) {
		int hash = getWordHash(word);
		while (true) {
			if (vocab_hash[hash] == -1) return -1;
			if (word.equals(vocab.get(vocab_hash[hash]).word))
				return vocab_hash[hash];
			hash = (hash + 1) % vocab_hash_size;
		}
	}

	public int addWordToVocab(String word) {
		Word w = new Word(word);
		vocab.add(w);
		vocab_size++;
		int hash = getWordHash(word);
		while (vocab_hash[hash] != -1)
			hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = vocab.size() - 1;
		return vocab.size() - 1;
	}

	public void sortVocab() {
		Collections.sort(vocab);
		for (int i = 0; i < vocab_hash_size; i++)
			vocab_hash[i] = -1;
		train_words = 0;
		ArrayList<Word> movs = new ArrayList<>();
		for (int i = 0; i < vocab.size(); i++) {
			if ((vocab.get(i).cn < min_count) && i != 0) {
				vocab_size--;
				movs.add(vocab.get(i));
			}
		}
		vocab.removeAll(movs);
		for (int i = 0; i < vocab.size(); i++) {
			Word w = vocab.get(i);
			int hash = getWordHash(w.word);
			while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
			vocab_hash[hash] = i;
			train_words += w.cn;
		}
	}

	public void reduceVocab() {
		System.out.println("Reduce Vocab Called");
		ArrayList<Word> movs = new ArrayList<>();
		for (Word w : vocab) {
			if (w.cn <= min_reduce)
				movs.add(w);
		}
		vocab.removeAll(movs);
		for (int i = 0; i < vocab_hash_size; i++)
			vocab_hash[i] = -1;
		for (int i = 0; i < vocab.size(); i++) {
			Word w = vocab.get(i);
			int hash = getWordHash(w.word);
			while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
			vocab_hash[hash] = i;
		}
		min_reduce++;
	}

	public void createBinaryTree() {
		long[] count = new long[vocab.size() * 2 + 1];
		long[] binary = new long[vocab.size() * 2 + 1];
		long[] parent_node = new long[vocab.size() * 2 + 1];
		char[] code = new char[max_code_length];
		long[] pointer = new long[max_code_length];

		for (int i = 0; i < vocab.size(); i++)
			count[i] = vocab.get(i).cn;
		for (int i = vocab.size(); i < vocab.size() * 2; i++)
			count[i] = (long) 1e15;
		int pos1 = vocab.size() - 1;
		int pos2 = vocab.size();
		int min1i, min2i;
		for (int i = 0; i < vocab.size() - 1; i++) {
			if (pos1 >= 0) {
				if (count[pos1] < count[pos2]) {
					min1i = pos1;
					pos1--;
				} else {
					min1i = pos2;
					pos2++;
				}
			} else {
				min1i = pos2;
				pos2++;
			}
			if (pos1 >= 0) {
				if (count[pos1] < count[pos2]) {
					min2i = pos1;
					pos1--;
				} else {
					min2i = pos2;
					pos2++;
				}
			} else {
				min2i = pos2;
				pos2++;
			}
			count[vocab.size() + i] = count[min1i] + count[min2i];
			parent_node[min1i] = vocab.size() + i;
			parent_node[min2i] = vocab.size() + i;
			binary[min2i] = 1;
		}

		for (int a = 0; a < vocab.size(); a++) {
			int b = a;
			int i = 0;
			while (true) {
				code[i] = (char) binary[b];
				pointer[i] = b;
				i++;
				b = (int) parent_node[b];
				if (b == vocab.size() * 2 - 2)
					break;
			}
			vocab.get(a).codeLen = (char) i;
			vocab.get(a).point[0] = vocab.size() - 2;
			for (b = 0; b < i; b++) {
				vocab.get(a).code[i - b - 1] = code[b];
				vocab.get(a).point[i - b] = (int) (pointer[b] - vocab.size());
			}
		}

        /*
		for (int a = 0; a < vocab.size(); a++){
            System.out.print(vocab.get(a).word+"'s code: ");
            for(int i = 0; i < vocab.get(a).codeLen; i++){
                System.out.print((int)vocab.get(a).code[i]);
            }
            System.out.println();
        }
        */
	}

	public void learnVocabFromTrainFile() throws IOException {
		for (int i = 0; i < vocab_hash_size; i++) vocab_hash[i] = -1;

		addWordToVocab("</s>");

		for (String line : lines) {
			String[] sentence = line.split("\\s");

			for (String word : sentence) {
				if (word.equalsIgnoreCase(""))
					continue;
				train_words++;
				int i = searchVocab(word);
				if (i == -1) {
					int a = addWordToVocab(word);
					vocab.get(a).cn = 1;
				} else
					vocab.get(i).cn++;

				if (vocab.size() > vocab_hash_size * 0.7)
					reduceVocab();
			}

			train_words++;
			String word = "</s>";
			int i = searchVocab(word);
			vocab.get(i).cn++;

		}
		sortVocab();
	}

	public void initNet() {
		syn0 = new float[vocab.size() * layer1_size];
		Random r = new Random();
		{
			sync1 = new float[vocab.size() * layer1_size];
			for (int i = 0; i < vocab.size(); i++)
				for (int j = 0; j < layer1_size; j++)
					sync1[i * layer1_size + j] = 0.0f;
		}

		{
			syn1neg = new float[vocab.size() * layer1_size];
			for (int i = 0; i < vocab.size(); i++)
				for (int j = 0; j < layer1_size; j++)
					syn1neg[i * layer1_size + j] = 0.0f;
		}

		for (int i = 0; i < vocab.size(); i++) {
			for (int j = 0; j < layer1_size; j++) {
				syn0[i * layer1_size + j] = (float) (r.nextFloat() - 0.5) / layer1_size;
			}
		}
		createBinaryTree();
	}

	/*
	 * @author: daidong
	 * Here, we try to init the vector[0] = location of blocks.
	 * This may help make sure that sequential accesses are well considered.
	 */
	public void reInitNet() {
		syn0 = new float[vocab.size() * layer1_size];
		Random r = new Random();
		{
			for (int i = 0; i < vocab.size(); i++)
				for (int j = 0; j < layer1_size; j++)
					sync1[i * layer1_size + j] = 0.0f;
		}

		{
			for (int i = 0; i < vocab.size(); i++)
				for (int j = 0; j < layer1_size; j++)
					syn1neg[i * layer1_size + j] = 0.0f;
		}

		for (int i = 0; i < vocab.size(); i++) {
			/*
			long blockId = 0;
			try {
				blockId = Long.parseLong(vocab.get(i).word);
			} catch (NumberFormatException e) {
				blockId = 0;
			}
			float location = (float) blockId / (float) (maxBlockId + 1);
			*/
			for (int j = 0; j < layer1_size; j++) {
				syn0[i * layer1_size + j] = (float) (r.nextFloat() - 0.5) / layer1_size;
				//syn0[i * layer1_size + j] = (float) (location - 0.5) / (float) layer1_size;
			}
		}
	}

	public void trainModel() throws IOException {
		long word, last_word;
		long word_count = 0, last_word_count = 0;
		Long sen[] = new Long[max_sentence_length];
		long target, label, local_iter = iter;
		long next_random = 5;
		float[] neu1 = new float[layer1_size];
		float[] neu1e = new float[layer1_size];


		while (true) {
			word_count_actual += word_count - last_word_count;
			word_count = 0;
			last_word_count = 0;
			int processed = 0;

			for (String line : lines) {

				//we assume each sentence is a single line.
				String[] sentence = line.split("\\s");
				processed = 0;
				while (processed < sentence.length) {

					if (word_count - last_word_count > 10000) {
						System.out.println("alpha: " + alpha + "\tProgress: " +
								(int) (word_count_actual / (double) (iter * train_words + 1) * 100) + "%");

						word_count_actual += (word_count - last_word_count);
						last_word_count = word_count;
						alpha = starting_alpha * (1 - word_count_actual / (float) (iter * train_words + 1));
						if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001f;
					}

					int sentence_length = 0;
					for (; processed < sentence.length && sentence_length < max_sentence_length; processed++) {
						String w = sentence[processed];
						word = searchVocab(w);
						if (word == -1) continue;
						if (word == 0) break;
						if (sample > 0) {
							double ran = (Math.sqrt(vocab.get((int) word).cn / (sample * train_words)) + 1) *
									(sample * train_words) / vocab.get((int) word).cn;
							next_random = next_random * 25214903917L + 11;
							if (ran < (next_random & 0xFFFF) / (float) 65536) continue;
						}
						sen[sentence_length++] = word;
					}
					word_count += sentence_length;

					for (int index = 0; index < sentence_length; index++) {

						word = sen[index];
						if (word == -1) continue;
						for (int c = 0; c < layer1_size; c++) neu1[c] = 0.0f;
						for (int c = 0; c < layer1_size; c++) neu1e[c] = 0.0f;
						next_random = next_random * 25214903917L + 11;
						int b = (int) next_random % window;

						int cw = 0, c;

						//input -> hidden
						for (int a = b; a < window * 2 + 1 - b; a++) {
							//for (int a = 0; a < window + 1; a++) {
							if (a != window) {
								c = index - window + a;
								if (c < 0) continue;
								if (c >= sentence_length) continue;
								last_word = sen[c];
								if (last_word == -1) continue;
								for (c = 0; c < layer1_size; c++)
									neu1[c] += syn0[((int) (c + last_word * layer1_size))];
								cw++;
							}
						}

						if (cw > 0) {

							for (c = 0; c < layer1_size; c++)
								neu1[c] /= cw;

							for (int d = 0; d < vocab.get((int) word).codeLen; d++) {
								float f = 0;
								long l2 = vocab.get((int) word).point[d] * layer1_size;

								//hidden -> output
								for (c = 0; c < layer1_size; c++) {
									f += neu1[c] * sync1[((int) (c + l2))];
								}
								if (f <= -MAX_EXP) continue;
								else if (f >= MAX_EXP) continue;
								else f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

								//g is gradient multiplied by the learning rate
								//float g = (1.0f - (float) vocab.get((int) word).code[d] - f) * alpha;
								float g = f * (1 - f) * (vocab.get((int) word).code[d] - f) * alpha;

								// Propagate errors output -> hidden
								for (c = 0; c < layer1_size; c++)
									neu1e[c] += g * sync1[((int) (c + l2))];
								// Learn weights hidden -> output
								for (c = 0; c < layer1_size; c++)
									sync1[((int) (c + l2))] += g * neu1[c];
							}

							//hidden -> in
							for (int a = b; a < window * 2 + 1 - b; a++) {
								if (a != window) {
									c = (int) (index - window + a);
									if (c < 0) continue;
									if (c >= sentence_length) continue;
									last_word = sen[c];
									if (last_word == -1) continue;
									for (c = 0; c < layer1_size; c++)
										syn0[((int) (c + last_word * layer1_size))] += neu1e[c];
								}
							}
						}
					}
				}
			}
			local_iter--;
			if (local_iter == 0) break;
		}
	}

	public void predictModel() throws IOException {
		int[] context = new int[window];
		for (int i = 0; i < window; i++) context[i] = -1;
		int context_length = 0;

		BufferedReader br = new BufferedReader(new FileReader(this.trainFile));
		String line;
		while ((line = br.readLine()) != null) {
			String[] fields = line.split(",");
			long ts = Long.parseLong(fields[0]);
			long offset = Long.parseLong(fields[4]);
			int size = Integer.parseInt(fields[5]);
			int response = Integer.parseInt(fields[6]);

			for (int i = context_length; i > 0; i--)
				context[i] = context[i - 1];

			long diskId = offset / 4096;
			int blocks = (int) Math.ceil((float) size / (float) 4096);

			//context[0] = diskId;

		}
	}


	public void trainModelWithTime() throws IOException {
		long word, last_word;
		long word_count = 0, last_word_count = 0;
		Long sen[] = new Long[max_sentence_length];
		Long time[] = new Long[max_sentence_length];

		long local_iter = iter;
		long next_random = 5;
		float[] neu1 = new float[layer1_size];
		float[] neu1e = new float[layer1_size];


		while (true) {
			word_count_actual += word_count - last_word_count;
			word_count = 0;
			last_word_count = 0;
			int processed = 0;

			//String line : lines
			for (int i = 0; i < lines.size(); i++) {
				String line = lines.get(i);
				String timeline = timelines.get(i);

				String[] sentence = line.split("\\s");
				String[] tss = timeline.split("\\s");

				processed = 0;
				while (processed < sentence.length) {

					if (word_count - last_word_count > 10000) {
						System.out.println("alpha: " + alpha + "\tProgress: " +
								(int) (word_count_actual / (double) (iter * train_words + 1) * 100) + "%");

						word_count_actual += (word_count - last_word_count);
						last_word_count = word_count;
						alpha = starting_alpha * (1 - word_count_actual / (float) (iter * train_words + 1));
						if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001f;
					}

					int sentence_length = 0;
					for (; processed < sentence.length && sentence_length < max_sentence_length; processed++) {
						String w = sentence[processed];
						String ts = tss[processed];

						word = searchVocab(w);
						if (word == -1) continue;
						if (word == 0) break;
						if (sample > 0) {
							double ran = (Math.sqrt(vocab.get((int) word).cn / (sample * train_words)) + 1) *
									(sample * train_words) / vocab.get((int) word).cn;
							next_random = next_random * 25214903917L + 11;
							if (ran < (next_random & 0xFFFF) / (float) 65536) continue;
						}
						sen[sentence_length] = word;
						time[sentence_length] = Long.parseLong(ts);

						sentence_length++;
					}
					word_count += sentence_length;


					//int local_time_window = time_sensitive_window;

					next_random = next_random * 25214903917L + 11;
					int local_time_window = (int) next_random % time_sensitive_window;

					for (int index = 0; index < sentence_length; index++) {

						word = sen[index];
						long ts = time[index];
						if (word == -1) continue;
						for (int c = 0; c < layer1_size; c++) neu1[c] = 0.0f;
						for (int c = 0; c < layer1_size; c++) neu1e[c] = 0.0f;

						int cw = 0, c;

						//input -> hidden
						for (int a = 0; a < window * 2 + 1; a++) {
							if (a != window) {
								c = index - window + a;
								if (c < 0) continue;
								if (c >= sentence_length) continue;
								last_word = sen[c];
								if (last_word == -1) continue;
								if (Math.abs(time[c] - ts) > local_time_window) continue;
								for (c = 0; c < layer1_size; c++)
									neu1[c] += syn0[((int) (c + last_word * layer1_size))];
								cw++;
							}
						}

						if (cw > 0) {

							for (c = 0; c < layer1_size; c++)
								neu1[c] /= cw;

							for (int d = 0; d < vocab.get((int) word).codeLen; d++) {
								float f = 0;
								long l2 = vocab.get((int) word).point[d] * layer1_size;

								//hidden -> output
								for (c = 0; c < layer1_size; c++) {
									f += neu1[c] * sync1[((int) (c + l2))];
								}
								if (f <= -MAX_EXP) continue;
								else if (f >= MAX_EXP) continue;
								else f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

								//g is gradient multiplied by the learning rate
								//float g = (1.0f - (float) vocab.get((int) word).code[d] - f) * alpha;
								float g = f * (1 - f) * (vocab.get((int) word).code[d] - f) * alpha;

								// Propagate errors output -> hidden
								for (c = 0; c < layer1_size; c++)
									neu1e[c] += g * sync1[((int) (c + l2))];
								// Learn weights hidden -> output
								for (c = 0; c < layer1_size; c++)
									sync1[((int) (c + l2))] += g * neu1[c];
							}

							//hidden -> in
							for (int a = 0; a < window * 2 + 1; a++) {
								if (a != window) {
									c = (int) (index - window + a);
									if (c < 0) continue;
									if (c >= sentence_length) continue;
									last_word = sen[c];
									if (last_word == -1) continue;
									if (Math.abs(time[c] - ts) > local_time_window) continue;
									for (c = 0; c < layer1_size; c++)
										syn0[((int) (c + last_word * layer1_size))] += neu1e[c];
								}
							}
						}
					}

				}
			}
			local_iter--;
			if (local_iter == 0) break;
		}
	}

	ArrayList<String> lines = new ArrayList<>();
	ArrayList<String> timelines = new ArrayList<>();
	long maxBlockId = 0L;
	HashMap<Long, Integer> sequencialMap = new HashMap<>();


	public void loadTrainFile() throws IOException {

		BufferedReader br = new BufferedReader(new FileReader(this.trainFile));
		String line;
		long previous = 0;
		StringBuilder sentenceSB = new StringBuilder();
		StringBuilder timelineSB = new StringBuilder();

		while ((line = br.readLine()) != null) {
			String[] fields = line.split(",");
			long ts = Long.parseLong(fields[0]);
			long offset = Long.parseLong(fields[4]);
			int size = Integer.parseInt(fields[5]);
			int response = Integer.parseInt(fields[6]);

			if (ts - previous > cutting_window && previous != 0) {
				lines.add(sentenceSB.toString());
				timelines.add(timelineSB.toString());
				sentenceSB = new StringBuilder();
				timelineSB = new StringBuilder();
			}

			long diskId = offset / 4096;

			/* Extract Large Block Accesses*/
			int blocks = (int) Math.ceil((float) size / (float) 4096);

			for (int i = 0; i < blocks; i++) {
				sentenceSB.append((diskId + i) + " ");
				timelineSB.append((ts + i) + " ");
			}

			if ((diskId + blocks) > maxBlockId)
				maxBlockId = (diskId + blocks);

			previous = ts;
		}

		br.close();
	}

	public void kmeans(int classes) throws IOException {
		System.out.println("Run KMeans on " + classes + " groups");
		int clcn = classes;
		int iter = 10, closeid;
		int centcn[] = new int[classes];
		ArrayList<Integer>[] clusters = new ArrayList[classes];
		for (int i = 0; i < classes; i++) clusters[i] = new ArrayList<Integer>();

		int cl[] = new int[(int) vocab_size];
		float closev, x;
		float cent[] = new float[classes * layer1_size];

		for (int a = 0; a < vocab_size; a++) {
			cl[a] = a % clcn;
		}

		for (int a = 0; a < iter; a++) {
			System.out.println("KMeans Progress: " + ((float) a / (float) iter) * 100 + " %");
			for (int b = 0; b < clcn * layer1_size; b++)
				cent[b] = 0;
			for (int b = 0; b < clcn; b++)
				centcn[b] = 1;
			for (int c = 0; c < vocab_size; c++) {
				for (int d = 0; d < layer1_size; d++)
					cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
				centcn[cl[c]]++;
			}
			for (int b = 0; b < clcn; b++) {
				closev = 0;
				for (int c = 0; c < layer1_size; c++) {
					cent[layer1_size * b + c] /= centcn[b];
					closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
				}
				closev = (float) Math.sqrt(closev);
				for (int c = 0; c < layer1_size; c++)
					cent[layer1_size * b + c] /= closev;
			}
			for (int c = 0; c < vocab_size; c++) {
				closev = -10;
				closeid = 0;
				for (int d = 0; d < clcn; d++) {
					x = 0;
					for (int b = 0; b < layer1_size; b++)
						x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
					if (x > closev) {
						closev = x;
						closeid = d;
					}
				}
				cl[c] = closeid;
			}
		}
		System.out.println("KMeans Progress: " + "100%");

		for (int c = 0; c < vocab_size; c++)
			clusters[cl[c]].add(c);

		BufferedWriter bw = new BufferedWriter(new FileWriter(this.outputFile + ".group" + classes));
		bw.write(vocab.size() + " " + layer1_size + "\n");
		for (int i = 0; i < classes; i++) {
			String outs = "";
			outs += ("class: " + i + ":\t{");
			for (int b : clusters[i]) {
				outs += (vocab.get(b).word + ",");
			}
			outs += "}\n";
			bw.write(outs);
		}
		bw.close();

		System.out.println("Group are persisted into: " + this.outputFile + ".group" + classes);

		int hit = 0;
		int unhit = 0;
		int total = 0;

		for (String line : lines) {
			String[] ioaccess = line.split("\\s");
			for (int i = 0; i < ioaccess.length; i++) {
				String io = ioaccess[i];
				int index = searchVocab(io);
				if (index == -1)
					continue;
				if (index == 0)
					continue;

				int g = cl[index];
				if ((i + 1) < ioaccess.length) {
					String next = ioaccess[i + 1];
					int nextInt = searchVocab(next);
					if (clusters[g].contains(nextInt)) {
						hit++;
					}
					total++;
				}
			}
		}
		System.out.println("Group: " + classes + " hit: " + hit + " total: " + total + " ratio: " + (double) hit /
				(double) total * 100 + " %");
	}


	public void outputTrainedBin() throws IOException {
		int a, b;
		CPrint p = new CPrint();

		//syn0 is the vector assign
		BufferedWriter bw = new BufferedWriter(new FileWriter(this.outputFile));
		bw.write(vocab.size() + " " + layer1_size + "\n");
		bw.close();
		if (binary) {
			for (a = 0; a < vocab.size(); a++) {
				float[] v = new float[layer1_size];
				for (b = 0; b < layer1_size; b++)
					v[((int) b)] = (float) syn0[((int) (a * layer1_size + b))];
				p.p(this.outputFile, vocab.get((int) a).word, v);
			}
		} else {
			bw = new BufferedWriter(new FileWriter(this.outputFile));
			for (a = 0; a < vocab.size(); a++) {
				bw.write(vocab.get((int) a).word + " ");
				for (b = 0; b < layer1_size; b++) {
					String v = String.format("%f ", syn0[((int) (a * layer1_size + b))]);
					bw.write(v);
				}
				bw.write("\n");
			}
			bw.close();
		}
		System.out.println("Finish output bin model, output: " + this.outputFile);
	}

	public void train() throws IOException {
		long a, b, c, d;
		System.out.println("Starting training using file " + this.trainFile);
		starting_alpha = alpha;

		loadTrainFile();

		learnVocabFromTrainFile();

		initNet();
		initUnigramTable();


		System.out.println("Train Model with Original Algorithm");
		trainModel();
		outputTrainedBin();
		for (int group = 1000; group <= 1000; group += 1000)
			kmeans(group);

		/*
		System.out.println("Train Model with Time Sensitive Algorithm");
		//alpha = 0.025f;
		//starting_alpha = alpha;
		//word_count_actual = 0;
		//reInitNet();
		trainModelWithTime();
		outputTrainedBin();
		for (int group = 1000; group <= 1000; group += 1000)
			kmeans(group);
			*/

	}

	public static void main(String[] args) throws IOException {
		Block2Vec instance = new Block2Vec(args[0], args[1]);
		instance.train();
	}
}