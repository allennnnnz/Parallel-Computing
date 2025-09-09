#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

static inline void swapf(float *a, float *b) { float t = *a; *a = *b; *b = t; }

/* 分散式驗證：全域是否已排序 (僅檢查當前記憶體中 local_data) */
static int verify_sorted(float *local_data, long local_n, int rank, int size, MPI_Comm comm) {
    int ok_local = 1;
    for (long i=0; i+1<local_n; ++i)
        if (local_data[i] > local_data[i+1]) { ok_local = 0; break; }

    float my_first = 0.f, my_last = 0.f;
    if (local_n > 0) { my_first = local_data[0]; my_last = local_data[local_n-1]; }

    // 邊界：傳遞每段最後一個給下一個
    if (rank < size-1) {
        MPI_Send(&my_last,1,MPI_FLOAT,rank+1,90,comm);
    }
    if (rank > 0) {
        float prev_last;
        MPI_Recv(&prev_last,1,MPI_FLOAT,rank-1,90,comm,MPI_STATUS_IGNORE);
        if (local_n > 0 && prev_last > my_first) ok_local = 0;
    }
    int ok_global;
    MPI_Allreduce(&ok_local,&ok_global,1,MPI_INT,MPI_LAND,comm);
    return ok_global;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if (argc < 4) {
        if (rank==0) fprintf(stderr,"Usage: %s n input.bin output.bin [--verify] [--metrics]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    long n = atol(argv[1]);
    const char *infile  = argv[2];
    const char *outfile = argv[3];

    // 旗標
    int opt_verify = 0, opt_metrics = 0;
    for (int i=4; i<argc; ++i) {
        if (strcmp(argv[i],"--verify")==0)  opt_verify = 1;
        else if (strcmp(argv[i],"--metrics")==0) opt_metrics = 1;
    }

    // 分配計算
    long base = (size>0)? (n/size):0;
    long extra = (size>0)? (n%size):0;
    long local_n = (rank < extra)? (base+1): base;

    float *local_data = NULL;
    if (local_n > 0) {
        local_data = (float*)malloc(local_n * sizeof(float));
        if (!local_data) { fprintf(stderr,"Rank %d malloc fail\n",rank); MPI_Abort(MPI_COMM_WORLD,2); }
    }

    MPI_Offset offset_elems;
    if (rank < extra)
        offset_elems = (MPI_Offset)rank * (base + 1);
    else
        offset_elems = (MPI_Offset)(extra * (base + 1) + (rank - extra) * base);

    int global_lastID = (int)(offset_elems + (local_n - 1));
    MPI_Offset file_offset_bytes = offset_elems * (MPI_Offset)sizeof(float);

    // 計時變數
    double t_total_start = MPI_Wtime();
    double t_read=0, t_sort=0, t_write=0;
    double t_comp=0, t_comm=0;          // 細分 (計算 vs 邊界通訊)
    long   comm_calls = 0;
    long   comm_bytes = 0;              // 計：雙向各 1 float => 2*sizeof(float) (方便表示總 wire bytes)

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // 讀檔
    MPI_File fh;
    int err = MPI_File_open(MPI_COMM_WORLD, infile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS) { if (rank==0) fprintf(stderr,"Open input fail\n"); MPI_Abort(MPI_COMM_WORLD,3); }
    if (local_n > 0) {
        err = MPI_File_read_at(fh, file_offset_bytes, local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
        if (err != MPI_SUCCESS) { fprintf(stderr,"Rank %d read fail\n",rank); MPI_Abort(MPI_COMM_WORLD,4); }
    }
    MPI_File_close(&fh);

    MPI_Barrier(MPI_COMM_WORLD);
    t_read = MPI_Wtime() - t0;

    // 排序
    double t_sort_start = MPI_Wtime();
    if (n > 1) {
        int global_changed = 1;
        bool odd_even;
        long phase;
        for (phase = 0; phase < n && global_changed; ++phase) {
            int local_changed = 0;

            int start;
            if (phase % 2 == 0) { odd_even = 0; start = 0; }
            else { odd_even = 1; start = 1; }

            double comp_start = MPI_Wtime();
            if (local_n > 1) {
                for (long i = start; i + 1 < local_n; i += 2) {
                    if (local_data[i] > local_data[i+1]) {
                        swapf(&local_data[i], &local_data[i+1]);
                        local_changed = 1;
                    }
                }
            }
            t_comp += MPI_Wtime() - comp_start;

            if (size > 1 && local_n > 0) {
                if ( (global_lastID % 2) == odd_even && rank + 1 < size) {
                    float send_val = local_data[local_n - 1], recv_val;
                    double cst = MPI_Wtime();
                    MPI_Sendrecv(&send_val, 1, MPI_FLOAT, rank + 1, 10,
                                 &recv_val, 1, MPI_FLOAT, rank + 1, 11,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    t_comm += MPI_Wtime() - cst;
                    comm_calls++;
                    comm_bytes += 2 * (long)sizeof(float);
                    if (send_val > recv_val) { local_data[local_n - 1] = recv_val; local_changed = 1; }
                }
                if ((global_lastID % 2) == odd_even && rank > 0 && local_n > 0) {
                    float send_val = local_data[0], recv_val;
                    double cst = MPI_Wtime();
                    MPI_Sendrecv(&send_val, 1, MPI_FLOAT, rank - 1, 11,
                                 &recv_val, 1, MPI_FLOAT, rank - 1, 10,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    t_comm += MPI_Wtime() - cst;
                    comm_calls++;
                    comm_bytes += 2 * (long)sizeof(float);
                    if (recv_val > send_val) { local_data[0] = recv_val; local_changed = 1; }
                }
            }
            MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        }
        // phases 實際使用可印 (phase 已自增結束)
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t_sort = MPI_Wtime() - t_sort_start;

    // 寫出
    MPI_Barrier(MPI_COMM_WORLD);
    double t_write_start = MPI_Wtime();
    err = MPI_File_open(MPI_COMM_WORLD, outfile,
                        MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if (err == MPI_SUCCESS) {
        if (local_n > 0)
            MPI_File_write_at(fh, file_offset_bytes, local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&fh);
    } else if (rank==0) {
        fprintf(stderr,"Open output fail\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t_write = MPI_Wtime() - t_write_start;

    // 驗證
    if (opt_verify) {
        int ok = verify_sorted(local_data, local_n, rank, size, MPI_COMM_WORLD);
        if (rank==0) printf("VERIFY: %s\n", ok? "OK":"FAILED");
    }

    double total_time = MPI_Wtime() - t_total_start;

    // 彙總 metrics
    if (opt_metrics) {
        double max_read, max_sort, max_write, max_total, max_comp, max_comm;
        MPI_Reduce(&t_read,&max_read,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&t_sort,&max_sort,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&t_write,&max_write,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&total_time,&max_total,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&t_comp,&max_comp,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&t_comm,&max_comm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        long sum_calls, sum_bytes;
        MPI_Reduce(&comm_calls,&sum_calls,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(&comm_bytes,&sum_bytes,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
        if (rank == 0) {
            const char *csvfile = "metrics.csv";

            // 檢查檔案是否存在
            FILE *fp_check = fopen(csvfile, "r");
            int write_header = 0;
            if (!fp_check) {
                write_header = 1; // 檔案不存在 → 需要寫 header
            } else {
                fclose(fp_check);
            }

            FILE *fp = fopen(csvfile, "a"); // 以 append 模式開啟
            if (!fp) { 
                fprintf(stderr, "Cannot open CSV file\n"); 
                MPI_Abort(MPI_COMM_WORLD, 5); 
            }

            if (write_header) {
                fprintf(fp, "n,size,read_max,sort_max,write_max,total_max,comp_max,comm_max,comm_calls_sum,comm_bytes_sum\n");
            }

            fprintf(fp, "%ld,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%ld,%ld\n",
                    n, size, max_read, max_sort, max_write, max_total, max_comp, max_comm, sum_calls, sum_bytes);

            fclose(fp);
        }
    }

    free(local_data);
    MPI_Finalize();
    return 0;
}