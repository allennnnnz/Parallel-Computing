#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <algorithm>
#define BOOST_SPREADSORT_MAX_SPLITS 12
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <nvtx3/nvToolsExt.h>
#include <cstring>
#include <numa.h>
#include <sched.h>

// 合併保留最小 k 個（輸出寫回 a）
inline void merge_keep_smallest_inplace(float *a, size_t n1,
                                        const float *b, size_t n2,
                                        float *buf, size_t k) {
    size_t i=0, j=0, t=0;
    while (t < k && i < n1 && j < n2)
        buf[t++] = (a[i] <= b[j]) ? a[i++] : b[j++];
    while (t < k && i < n1) buf[t++] = a[i++];
    while (t < k && j < n2) buf[t++] = b[j++];
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

    int cpu = sched_getcpu();
    int numa_node = numa_node_of_cpu(cpu);
    float *local_data = (float*)numa_alloc_onnode(sizeof(float) * local_n, numa_node);

    MPI_Offset offset_elems = (rank < extra) ? rank * (base + 1) :
                              (extra * (base + 1) + (rank - extra) * base);
    MPI_Offset file_offset_bytes = offset_elems * sizeof(float);

    // 讀取資料
    MPI_File fh;
    nvtxRangePop(); nvtxRangePush("I/O");
    MPI_File_open(MPI_COMM_SELF, infile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at(fh, file_offset_bytes, local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    nvtxRangePop(); nvtxRangePush("CPU");

    // 用 boost spreadsort 排序
    boost::sort::spreadsort::spreadsort(local_data, local_data + local_n);

    // 預先分配最大可能的 neighbor buffer 和 merge buffer
    long max_neighbor_n = base + 1;
    float *neighbor_data = (float*)numa_alloc_onnode(sizeof(float) * max_neighbor_n, numa_node);
    float *merge_buf = (float*)numa_alloc_onnode(sizeof(float) * local_n, numa_node);

    if(local_n == 0){
        MPI_Barrier(MPI_COMM_WORLD);
        numa_free(local_data, sizeof(float) * local_n);
        numa_free(neighbor_data, sizeof(float) * max_neighbor_n);
        numa_free(merge_buf, sizeof(float) * local_n);
        MPI_Finalize();
        return 0;
    }

    for(int phase = 0; phase < size+1; phase++){
        bool i_am_left = (phase % 2 == 0) ? rank % 2 == 0 : rank % 2 == 1;
        int peer_rank = i_am_left ? rank + 1 : rank - 1;
        long neighbor_n = (peer_rank < extra) ? (base + 1) : base;
        // 邊界檢查
        if ( neighbor_n <= 0 || (i_am_left && rank + 1 >= size) || (!i_am_left && rank - 1 < 0))
            continue;

        // 先傳邊界值
        float my_boundary = i_am_left ? local_data[local_n-1] : local_data[0];
        float neighbor_boundary = 0.0f;

        MPI_Sendrecv(&my_boundary, 1, MPI_FLOAT, peer_rank, 100,
                    &neighbor_boundary, 1, MPI_FLOAT, peer_rank, 100,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        bool need_exchange = i_am_left ? (my_boundary > neighbor_boundary)
                                       : (my_boundary < neighbor_boundary);

        if (!need_exchange)
            continue;

        // MPI 通訊整個 chunk
        nvtxRangePop(); nvtxRangePush("Comm");
        MPI_Sendrecv(local_data, local_n, MPI_FLOAT, peer_rank, phase % 2,
                    neighbor_data, neighbor_n, MPI_FLOAT, peer_rank, phase % 2,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        nvtxRangePop(); nvtxRangePush("CPU");

        // 合併資料
        if (i_am_left) {
            merge_keep_smallest_inplace(local_data, local_n,
                                        neighbor_data, neighbor_n,
                                        merge_buf, local_n);
        } else {
            merge_keep_largest_inplace(local_data, local_n,
                                       neighbor_data, neighbor_n,
                                       merge_buf, local_n);
        }
        std::swap(local_data, merge_buf);
    }

    nvtxRangePop(); nvtxRangePush("IO");
    MPI_File_open(MPI_COMM_SELF, outfile, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_write_at(fh, file_offset_bytes, local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    nvtxRangePop();

    nvtxRangePush("Comm");
    MPI_Barrier(MPI_COMM_WORLD);
    nvtxRangePop();

    numa_free(local_data, sizeof(float) * local_n);
    numa_free(neighbor_data, sizeof(float) * max_neighbor_n);
    numa_free(merge_buf, sizeof(float) * local_n);
    MPI_Finalize();
    return 0;
}