#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

static inline void swapf(float *a, float *b) {
    float t = *a; *a = *b; *b = t;
}

int main(int argc, char* argv[]) {
    //初始化
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);//process編號
    MPI_Comm_size(MPI_COMM_WORLD,&size);//process數

    //讀入參數處理
    if (argc != 4) {
        if (rank==0) fprintf(stderr,"Usage: %s n input.bin output.bin\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    long n = atol(argv[1]);
    const char *infile  = argv[2];
    const char *outfile = argv[3];

    //每個process要處理多少資料
    long base;
    long extra;
    long local_n;

    // 確保 process 數量大於 0
    if (size > 0) {
        base  = n / size;  // 每個 rank 至少分到的數量
        extra = n % size;  // 前 extra 個 rank 多拿 1 個
    } else {
        base  = 0;
        extra = 0;
    }

    // 計算這個 rank 要拿幾個元素
    if (rank < extra) {
        local_n = base + 1;   // 前 extra 個 rank 多拿 1
    } else {
        local_n = base;       // 其他 rank 拿 base 個
    }

    float *local_data = NULL;
    if (local_n > 0) {
        local_data = (float*)malloc(local_n * sizeof(float));
        if (!local_data) { fprintf(stderr,"Rank %d malloc fail\n",rank); MPI_Abort(MPI_COMM_WORLD,2); }
    }

    // 元素位移
    MPI_Offset offset_elems;
    //前extra的element會處理多一個資料
    if (rank < extra)
        offset_elems = (MPI_Offset)rank * (base + 1);
    else
        // 加上前面那些比較多的element數為基準
        offset_elems = (MPI_Offset)(extra * (base + 1) + (rank - extra) * base);

    int global_lastID = offset_elems + (local_n - 1);
    // mpi offset以long為單位所以要調整對應的offset位置
    MPI_Offset file_offset_bytes = offset_elems * (MPI_Offset)sizeof(float);

    // 讀檔
    MPI_File fh;
    int err = MPI_File_open(MPI_COMM_WORLD, infile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS) { if (rank==0) fprintf(stderr,"Open input fail\n"); MPI_Abort(MPI_COMM_WORLD,3); }
    if (local_n > 0) {
        err = MPI_File_read_at(fh, file_offset_bytes, local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
        if (err != MPI_SUCCESS) { fprintf(stderr,"Rank %d read fail\n",rank); MPI_Abort(MPI_COMM_WORLD,4); }
    }
    MPI_File_close(&fh);

    // 全域 odd-even transposition sort
    //把odd even分別都看成一個iter
    if (n > 1) {
        int global_changed = 1;
        // 最多做n次且任一rank都還有發生swap
        bool odd_even;// even = 0 odd = 1
        for (long phase = 0; phase < n && global_changed; ++phase) {
            int local_changed = 0;

            // 1. 本地比較 (根據 phase 決定起始 index)

            // 用phase來分odd even如果是偶數iter就是even基數就是odd
            int start;
            if (phase % 2 == 0) {
                odd_even = 0;
                start = 0;   // 如果 phase 是偶數 (even-phase)，從 index=0 開始比較
            } else {
                odd_even = 1;
                start = 1;   // 如果 phase 是奇數 (odd-phase)，從 index=1 開始比較
            }
            // 做自己chunk內的swap
            if (local_n > 1) {
                for (long i = start; i + 1 < local_n; i += 2) {
                    if (local_data[i] > local_data[i+1]) {
                        swapf(&local_data[i], &local_data[i+1]);
                        local_changed = 1;
                    }
                }
            }

            // 2. 邊界交換
            if (size > 1 && local_n > 0) {
                // even 左側最後一個要跟右側第一個交換，並由左側負責
                // odd  右側最後一個要跟左側第一個交換，並由左側負責
                if ( (global_lastID % 2) == odd_even && rank + 1 < size) {
                    float send_val = local_data[local_n - 1];
                    float recv_val;
                    MPI_Sendrecv(&send_val, 1, MPI_FLOAT, rank + 1, 10,
                                 &recv_val, 1, MPI_FLOAT, rank + 1, 11,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (send_val > recv_val) {
                        local_data[local_n - 1] = recv_val; // 保留較小
                        local_changed = 1;
                    }
                }
                // 右側 complementary 與左鄰
                if ((global_lastID % 2) == odd_even && rank > 0 && local_n > 0) {
                    float send_val = local_data[0];
                    float recv_val;
                    MPI_Sendrecv(&send_val, 1, MPI_FLOAT, rank - 1, 11,
                                 &recv_val, 1, MPI_FLOAT, rank - 1, 10,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (recv_val > send_val) {
                        local_data[0] = recv_val; // 保留較大
                        local_changed = 1;
                    }
                }
            }

            // 3. 全域是否仍需繼續 只要有任意一個 rank 的 local_changed=1，最後 global_changed 就會是 1。
            MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        }
    }

    // 寫出
    err = MPI_File_open(MPI_COMM_WORLD, outfile,
                        MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if (err == MPI_SUCCESS) {
        if (local_n > 0)
            MPI_File_write_at(fh, file_offset_bytes, local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&fh);
    } else if (rank==0) {
        fprintf(stderr,"Open output fail\n");
    }

    free(local_data);
    MPI_Finalize();
    return 0;
}