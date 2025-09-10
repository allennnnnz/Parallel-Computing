#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 寫 float 陣列到二進位檔
void write_floats(const char *filename, float *arr, int n) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror("fopen");
        exit(1);
    }
    fwrite(arr, sizeof(float), n, f);
    fclose(f);
}

// 比較函數給 qsort 用
int cmp_float(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb); // fa>fb →1, fa<fb→-1
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("用法: %s <n1> <n2> ...\n", argv[0]);
        return 1;
    }

    srand((unsigned)time(NULL));

    for (int argi = 1; argi < argc; argi++) {
        int n = atoi(argv[argi]);
        if (n <= 0) {
            fprintf(stderr, "n 必須是正整數: %s\n", argv[argi]);
            continue;
        }

        float *data = malloc(n * sizeof(float));
        if (!data) {
            perror("malloc");
            return 1;
        }

        // 產生隨機 float (-100.0 ~ 100.0)
        for (int i = 0; i < n; i++) {
            data[i] = ((float)rand() / RAND_MAX) * 200.0f - 100.0f;
        }

        // 輸出檔名 (包含資料大小)
        char input_name[256], output_name[256];
        snprintf(input_name, sizeof(input_name), "input_%d.bin", n);
        snprintf(output_name, sizeof(output_name), "expected_output_%d.bin", n);

        // 寫 input.bin
        write_floats(input_name, data, n);

        // 複製並排序
        float *sorted = malloc(n * sizeof(float));
        if (!sorted) {
            perror("malloc sorted");
            free(data);
            return 1;
        }
        for (int i = 0; i < n; i++) sorted[i] = data[i];
        qsort(sorted, n, sizeof(float), cmp_float);

        // 寫 expected_output.bin
        write_floats(output_name, sorted, n);

        printf("已產生 %s 和 %s\n", input_name, output_name);

        free(data);
        free(sorted);
    }

    return 0;
}