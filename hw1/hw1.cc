#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <vector>
#include <algorithm>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <nvtx3/nvToolsExt.h>
#include <cstring>

// 合併保留最小 k 個（輸出寫回 a）
inline void merge_keep_smallest_inplace(float *a, size_t n1,
                                        const float *b, size_t n2,
                                        float *buf, size_t k) {
    size_t i=0, j=0, t=0;
    while (t < k && i < n1 && j < n2)
        buf[t++] = (a[i] <= b[j]) ? a[i++] : b[j++];
    while (t < k && i < n1) buf[t++] = a[i++];
    while (t < k && j < n2) buf[t++] = b[j++];
    std::memcpy(a, buf, k * sizeof(float));
}

// 合併保留最大 k 個（輸出寫回 a）
inline void merge_keep_largest_inplace(float *a, size_t n1,
                                       const float *b, size_t n2,
                                       float *buf, size_t k) {
    size_t i=n1, j=n2;
    size_t t=k;
    while (t>0 && i>0 && j>0)
        buf[--t] = (a[i-1] >= b[j-1]) ? a[--i] : b[--j];
    while (t>0 && i>0) buf[--t] = a[--i];
    while (t>0 && j>0) buf[--t] = b[--j];
    std::memcpy(a, buf, k*sizeof(float));
}

int main(int argc, char* argv[]) {
    nvtxRangePush("CPU");
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    long n = atol(argv[1]);
    const char *infile  = argv[2];
    const char *outfile = argv[3];

    long base = n / size;
    long extra = n % size;
    long local_n = (rank < extra)? (base+1): base;

    std::vector<float> local_data_vec(local_n);

    MPI_Offset offset_elems = (rank < extra) ? rank * (base + 1) :
                              (extra * (base + 1) + (rank - extra) * base);
    MPI_Offset file_offset_bytes = offset_elems * sizeof(float);

    // 讀取資料
    MPI_File fh;
    nvtxRangePop(); nvtxRangePush("I/O");
    MPI_File_open(MPI_COMM_SELF, infile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at(fh, file_offset_bytes, local_data_vec.data(), local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    nvtxRangePop(); nvtxRangePush("CPU");

    boost::sort::spreadsort::spreadsort(local_data_vec.begin(), local_data_vec.end());

    // 預先分配最大可能的 neighbor buffer 和 merge buffer
    std::vector<float> neighbor_data_vec(base+1);
    std::vector<float> merge_buf(local_n + base + 1);

    for(int phase = 0; phase < size+1; phase++){
        if(phase % 2 == 0){ // 偶數 phase
            if(rank % 2 == 0 && rank + 1 < size){
                long neighbor_n = (rank + 1 < extra) ? (base+1) : base;

                nvtxRangePop(); nvtxRangePush("Comm");
                MPI_Sendrecv(local_data_vec.data(), local_n, MPI_FLOAT, rank + 1, 0,
                             neighbor_data_vec.data(), neighbor_n, MPI_FLOAT, rank + 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                nvtxRangePop(); nvtxRangePush("CPU");

                merge_keep_smallest_inplace(local_data_vec.data(), local_n,
                                            neighbor_data_vec.data(), neighbor_n,
                                            merge_buf.data(), local_n);
            } else if(rank % 2 == 1){
                long neighbor_n = (rank - 1 < extra) ? (base+1) : base;

                nvtxRangePop(); nvtxRangePush("Comm");
                MPI_Sendrecv(local_data_vec.data(), local_n, MPI_FLOAT, rank - 1, 0,
                             neighbor_data_vec.data(), neighbor_n, MPI_FLOAT, rank - 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                nvtxRangePop(); nvtxRangePush("CPU");

                merge_keep_largest_inplace(local_data_vec.data(), local_n,
                                           neighbor_data_vec.data(), neighbor_n,
                                           merge_buf.data(), local_n);
            }
        } else { // 奇數 phase
            if(rank % 2 == 1 && rank + 1 < size){
                long neighbor_n = (rank + 1 < extra) ? (base+1) : base;

                nvtxRangePop(); nvtxRangePush("Comm");
                MPI_Sendrecv(local_data_vec.data(), local_n, MPI_FLOAT, rank + 1, 1,
                             neighbor_data_vec.data(), neighbor_n, MPI_FLOAT, rank + 1, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                nvtxRangePop(); nvtxRangePush("CPU");

                // 奇數 phase 中，奇數 rank 是較低索引，需保留較小的一半
                merge_keep_smallest_inplace(local_data_vec.data(), local_n,
                                           neighbor_data_vec.data(), neighbor_n,
                                           merge_buf.data(), local_n);
            } else if(rank % 2 == 0 && rank > 0){
                long neighbor_n = (rank - 1 < extra) ? (base+1) : base;

                nvtxRangePop(); nvtxRangePush("Comm");
                MPI_Sendrecv(local_data_vec.data(), local_n, MPI_FLOAT, rank - 1, 1,
                             neighbor_data_vec.data(), neighbor_n, MPI_FLOAT, rank - 1, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                nvtxRangePop(); nvtxRangePush("CPU");

                // 偶數（非 0）為較高索引，需保留較大的一半
                merge_keep_largest_inplace(local_data_vec.data(), local_n,
                                            neighbor_data_vec.data(), neighbor_n,
                                            merge_buf.data(), local_n);
            }
        }
    }

    nvtxRangePop(); nvtxRangePush("IO");
    MPI_File_open(MPI_COMM_SELF, outfile, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_write_at(fh, file_offset_bytes, local_data_vec.data(), local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    // 結束 I/O 區段
    nvtxRangePop();
    nvtxRangePush("Comm");
    MPI_Barrier(MPI_COMM_WORLD);
    // 結束 Comm 區段
    nvtxRangePop();
    MPI_Finalize();
    return 0;
}