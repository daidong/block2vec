#include <jni.h>
#include <stdio.h>
#include "utils_CPrint.h"

JNIEXPORT void JNICALL Java_utils_CPrint_cprint
  (JNIEnv *env, jobject thisobj, jstring fileJNIStr, jstring wordJNIStr, jfloatArray floatJNIArray){

    const char *fileCStr = (*env)->GetStringUTFChars(env, fileJNIStr, NULL);

    const char *wordCStr = (*env)->GetStringUTFChars(env, wordJNIStr, NULL);

    jfloat *floatCArray = (*env)->GetFloatArrayElements(env, floatJNIArray, NULL);

    jsize length = (*env)->GetArrayLength(env, floatJNIArray);

    FILE *fo = fopen(fileCStr, "a+");
    fprintf(fo, "%s ", wordCStr);
    for (int b = 0; b < length; b++){
        fwrite(&floatCArray[b], sizeof(float), 1, fo);
    }
    fprintf(fo, "\n");
    fclose(fo);
}