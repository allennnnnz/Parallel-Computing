#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
 #include <stdbool.h>

int swap(float* a, float* b) {
    float temp = *a;
    *a = *b;
    *b = temp;
    return 1;
}

void earlysyop(bool odd_swap,bool even_swap){
    if((odd_swap & even_swap)==0){
        MPI_File fh;
        MPI_File_open(MPI_COMM_WORLD, outfile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        MPI_File_write_at(fh, readOffset, local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&fh);
        free(local_data);
        MPI_Finalize();
        return 0;
    }
}

int exchange_with_neighbors(float *local_data, int local_n,
                            int rank, int size, int phase) {
    int local_swap = 0;
    float neighbor;

    if (phase == 0) {
        // odd phase: 奇數 rank 和右邊交換，最後一個rank沒有右鄰居
        // send_val我的最後一個
        // neighbor別人的第一個
        if (rank % 2 == 1 && rank + 1 < size) {
            float send_val = local_data[local_n - 1];
            MPI_Sendrecv(&send_val, 1, MPI_FLOAT, rank+1, 0,
                         &neighbor, 1, MPI_FLOAT, rank+1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (send_val > neighbor) {
                local_data[local_n - 1] = neighbor;
                local_swap = 1;
            }
        }
        // 奇數 rank 的右邊 process (rank+1) 要更新 local_data[0]，第一個rank沒有左鄰居
        if (rank % 2 == 0 && rank > 0) {
            float send_val = local_data[0];
            MPI_Sendrecv(&send_val, 1, MPI_FLOAT, rank-1, 0,
                         &neighbor, 1, MPI_FLOAT, rank-1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (neighbor > send_val) {
                local_data[0] = neighbor;
                local_swap = 1;
            }
        }
    } else {
        // even phase: 偶數 rank 和右邊交換
        if (rank % 2 == 0 && rank + 1 < size) {
            float send_val = local_data[local_n - 1];
            MPI_Sendrecv(&send_val, 1, MPI_FLOAT, rank+1, 0,
                         &neighbor, 1, MPI_FLOAT, rank+1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (send_val > neighbor) {
                local_data[local_n - 1] = neighbor;
                local_swap = 1;
            }
        }
        if (rank % 2 == 1) {
            float send_val = local_data[0];
            MPI_Sendrecv(&send_val, 1, MPI_FLOAT, rank-1, 0,
                         &neighbor, 1, MPI_FLOAT, rank-1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (neighbor > send_val) {
                local_data[0] = neighbor;
                local_swap = 1;
            }
        }
    }
    return local_swap;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    // Step 0: MPI 初始化
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) {
            printf("Usage: %s n input.bin output.bin\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Step 1: parse arguments
    long n = atol(argv[1]);
    char* infile = argv[2];
    char* outfile = argv[3];

    // Step 2: compute local partition (每個 process 負責多少元素)
    // local n代表我要負責幾份data
    // 餘數平均分給前面幾個process 負責的 chunk
    long local_n = n / size;
    long remainder = n % size;
    if (rank < remainder) {
        local_n++;
    }

    // Step 3: allocate local buffer
    float* local_data = (float*)malloc(local_n * sizeof(float));

    // Step 4: 用 MPI_File_read_at 讀 input.bin
    MPI_Offset readOffset;

    if (rank < remainder) {
        readOffset = rank * local_n * sizeof(float);
    }else {
        readOffset = remainder * (local_n + 1) * sizeof(float) + (rank - remainder) * local_n * sizeof(float);
    }

    // 本chunk中最後一個element的gID是多少
    long global_lastID = readOffset + local_n;

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, infile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at(fh, readOffset, local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    
    // swap done flag
    bool odd_swap = 0;
    bool even_swap = 0;
    // even phase 
    // local swap
    for(int l = 0 ; l<totle_n ; l++){
        for(int i=0 ; i+1 <local_n ; i+=2){
        if(local_data[i] > local_data[i+1]){
            swap(&local_data[i],&local_data[i+1])
            even_swap = 1;
            }
        }
        //如果最後一個index是偶數就要跟鄰居交換
        if (global_lastID % 2 == 0)
        {
            even_swap |= exchange_with_neighbors();
        }
        // odd phase
        for(int i=1 ; i+1 <local_n ; i+=2){
            if(local_data[i] > local_data[i+1])
                swap(&local_data[i],&local_data[i+1])
                odd_swap = 1;
        }
        //如果最後一個index是偶數就要跟鄰居交換
        if (global_lastID != 2 == 0)
        {
            odd_swap |= exchange_with_neighbors();
        }
        earlysyop(odd_swap,even_swap);
    }
    
    earlystop();
}