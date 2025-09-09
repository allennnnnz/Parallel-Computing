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

// 讀 float 陣列（方便檢查）
void read_floats(const char *filename, int n) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("fopen");
        exit(1);
    }
    float val;
    printf("%s: [", filename);
    for (int i = 0; i < n; i++) {
        fread(&val, sizeof(float), 1, f);
        printf("%f", val);
        if (i != n - 1) printf(", ");
    }
    printf("]\n");
    fclose(f);
}

// 比較函數給 qsort 用
int cmp_float(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb); // fa>fb →1, fa<fb→-1
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("用法: %s <n>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    float *data = malloc(n * sizeof(float));
    if (!data) {
        perror("malloc");
        return 1;
    }

    srand((unsigned)time(NULL));

    // 產生隨機 float (-100.0 ~ 100.0)
    for (int i = 0; i < n; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 200.0f - 100.0f;
    }

    // 寫 input.bin
    write_floats("input.bin", data, n);

    // 複製並排序
    float *sorted = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) sorted[i] = data[i];
    qsort(sorted, n, sizeof(float), cmp_float);

    // 寫 expected_output.bin
    write_floats("output.bin", sorted, n);

    printf("已產生 input.bin 和 expected_output.bin\n");
    read_floats("input.bin", n);
    read_floats("expected_output.bin", n);

    free(data);
    free(sorted);
    return 0;
}