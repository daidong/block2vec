package utils;

/**
 * Created by daidong on 2/19/16.
 */
public class CPrint {

    static {
        System.loadLibrary("cfloat");
    }

    private native void cprint(String file, String word, float[] data);

    public void p(String file, String word, float[] data){
        cprint(file, word, data);
    }

    public static void main(String[] args){
        CPrint p = new CPrint();
        float[] t = new float[2];
        t[0] = 0.1f;
        t[1] = 0.2f;

        p.p("/tmp/a.txt", "word", t);
    }
}
