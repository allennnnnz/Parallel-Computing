#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

int cmp_float(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

/* 驗證排序正確性（檢查局部 + 邊界） */
static int verify_sorted(float *local_data, long local_n, int rank, int size, MPI_Comm comm) {
    int ok_local = 1;
    for (long i=0; i+1<local_n; ++i)
        if (local_data[i] > local_data[i+1]) { ok_local = 0; break; }

    float my_first = (local_n > 0) ? local_data[0] : INFINITY;
    float my_last  = (local_n > 0) ? local_data[local_n-1] : -INFINITY;

    // 與左鄰居比較
    if (rank > 0) {
        float prev_last;
        MPI_Sendrecv(&my_first, 1, MPI_FLOAT, rank-1, 100,
                     &prev_last, 1, MPI_FLOAT, rank-1, 100,
                     comm, MPI_STATUS_IGNORE);
        if (prev_last > my_first) ok_local = 0;
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

    int opt_verify = 0, opt_metrics = 0;
    for (int i=4; i<argc; ++i) {
        if (strcmp(argv[i],"--verify")==0)  opt_verify = 1;
        else if (strcmp(argv[i],"--metrics")==0) opt_metrics = 1;
    }

    // 分配 local chunk
    long base = (n/size);
    long extra = (n%size);
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
    MPI_Offset file_offset_bytes = offset_elems * (MPI_Offset)sizeof(float);

    // 計時變數
    double t_read=0, t_write=0, t_comp=0, t_comm=0;

    // ===== 讀檔 =====
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
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

    // ===== 排序 =====
    // 1. 初始 local quicksort
    if (local_n > 1) {
        double comp_start = MPI_Wtime();
        qsort(local_data, local_n, sizeof(float), cmp_float);
        t_comp += MPI_Wtime() - comp_start;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // 2. 邊界交換直到全域有序
    int global_changed = 1;
    while (global_changed) {
        int local_changed = 0;

        // 與右鄰居比較
        if (rank < size-1 && local_n > 0) {
            float my_max = local_data[local_n-1];
            float neighbor_min;
            double cst = MPI_Wtime();
            MPI_Sendrecv(&my_max, 1, MPI_FLOAT, rank+1, 200,
                         &neighbor_min, 1, MPI_FLOAT, rank+1, 200,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            t_comm += MPI_Wtime() - cst;

            if (my_max > neighbor_min) {
                // 更新本地邊界
                local_data[local_n-1] = neighbor_min;
                local_changed = 1;
            }
        }

        // 與左鄰居比較
        if (rank > 0 && local_n > 0) {
            float my_min = local_data[0];
            float neighbor_max;
            double cst = MPI_Wtime();
            MPI_Sendrecv(&my_min, 1, MPI_FLOAT, rank-1, 300,
                         &neighbor_max, 1, MPI_FLOAT, rank-1, 300,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            t_comm += MPI_Wtime() - cst;

            if (neighbor_max > my_min) {
                local_data[0] = neighbor_max;
                local_changed = 1;
            }
        }

        // 若有更新，再次 quicksort
        if (local_changed && local_n > 1) {
            double comp_start = MPI_Wtime();
            qsort(local_data, local_n, sizeof(float), cmp_float);
            t_comp += MPI_Wtime() - comp_start;
        }

        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    }

    // ===== 寫檔 =====
    MPI_Barrier(MPI_COMM_WORLD);
    double t_write_start = MPI_Wtime();
    err = MPI_File_open(MPI_COMM_WORLD, outfile, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if (err == MPI_SUCCESS) {
        if (local_n > 0)
            MPI_File_write_at(fh, file_offset_bytes, local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&fh);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t_write = MPI_Wtime() - t_write_start;

    // ===== 驗證 =====
    if (opt_verify) {
        int ok = verify_sorted(local_data, local_n, rank, size, MPI_COMM_WORLD);
        if (rank==0) printf("VERIFY: %s\n", ok? "OK":"FAILED");
    }

    // ===== 彙總 metrics =====
    if (opt_metrics) {
        double max_read, max_write, max_comp, max_comm;
        MPI_Reduce(&t_read,&max_read,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&t_write,&max_write,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&t_comp,&max_comp,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(&t_comm,&max_comm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

        if (rank == 0) {
            const char *csvfile = "metrics.csv";
            FILE *fp_check = fopen(csvfile, "r");
            int write_header = 0;
            if (!fp_check) write_header = 1;
            else fclose(fp_check);

            FILE *fp = fopen(csvfile, "a");
            if (!fp) { fprintf(stderr, "Cannot open CSV file\n"); MPI_Abort(MPI_COMM_WORLD, 5); }

            if (write_header) {
                fprintf(fp, "n,size,io_max,comp_max,comm_max,total_max\n");
            }
            double io_max = max_read + max_write;
            double total_max = io_max + max_comp + max_comm;

            fprintf(fp, "%ld,%d,%.6f,%.6f,%.6f,%.6f\n",
                    n, size, io_max, max_comp, max_comm, total_max);

            fclose(fp);
        }
    }

    free(local_data);
    MPI_Finalize();
    return 0;
}