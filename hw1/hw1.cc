// 程式說明：
// 使用 MPI 進行平行 odd-even transposition sort（相鄰交換排序）。
// 每個 rank 先各自對區塊做本地排序，之後在奇偶 phase 中與相鄰 rank 交換資料並合併，
// 低 rank 保留較小的一半，高 rank 保留較大的一半。以 MPI_Allreduce 偵測是否還有變動來終止。

#include <mpi.h>              // MPI 通訊與 MPI-IO
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <vector>             // 動態陣列容器
#include <algorithm>          // 標準演算法（本檔主要用 boost 的 sort）
#include <cstring>
#include <boost/sort/spreadsort/spreadsort.hpp> // Boost spreadsort：對浮點數有不錯效能
#include <nvtx3/nvToolsExt.h> // NVTX 標註，方便在 Nsight 中觀察區段



// 手寫線性合併：兩個已排序區間 a(長度 n1)、b(長度 n2) -> out (長度 n1+n2)
// 回傳值：若 out 與 a、b 的原內容不同，則 return 1；否則 return 0
// 用於 odd-even 交換後的雙向合併，再由低/高 rank 擷取各自需要的一半。
static inline void merge_two(const float *a, size_t n1,
                            const float *b, size_t n2,
                            float *out) {
    size_t i=0, j=0, k=0;
    

    while (i < n1 && j < n2) {
        float val;
        if (a[i] <= b[j]) {
            val = a[i++];
        } else {
            val = b[j++];
        }
       
        out[k++] = val;
    }

    while (i < n1) {
        
        out[k++] = a[i++];
    }

    while (j < n2) {
        
        out[k++] = b[j++];
    }
}

int main(int argc, char* argv[]) {
    // 初始化 MPI 與 NVTX 標註
    nvtxRangePush("CPU");
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    // 讀取參數：n(元素個數)、輸入檔、輸出檔
    long n = atol(argv[1]);
    const char *infile  = argv[2];
    const char *outfile = argv[3];
    
    // 依據 n 與 process 數分配每個 rank 的工作量（前 extra 個 rank 會多 1 個元素）
    long base = (n/size);
    long extra = (n%size);
    long local_n = (rank < extra)? (base+1): base;

    // 如果資料量小於 process 數量，改用單機排序
    

    // 每個 rank 的本地緩衝區
    std::vector<float> local_data_vec(local_n);

    // 計算每個 rank 在檔案中的起始偏移（以元素為單位再轉成位元組）
    MPI_Offset offset_elems;
    if (rank < extra)
        offset_elems = (MPI_Offset)rank * (base + 1);
    else
        offset_elems = (MPI_Offset)(extra * (base + 1) + (rank - extra) * base);
    MPI_Offset file_offset_bytes = offset_elems * (MPI_Offset)sizeof(float);

    // 利用 MPI-IO 讀取各自的資料區段
    MPI_File fh;
    nvtxRangePop();nvtxRangePush("I/O");
    MPI_File_open(MPI_COMM_SELF, infile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at(fh, file_offset_bytes, local_data_vec.data(), local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    nvtxRangePop();nvtxRangePush("CPU");
    // 此時 local_data_vec 已經載入本 rank 的資料

    // 先做本地排序（使用 boost::spreadsort 對浮點數速度佳）
    
    boost::sort::spreadsort::spreadsort(local_data_vec.begin(), local_data_vec.end());



    std::vector<float> neighbor_data_vec;
    std::vector<float> combined_data;

    // 進行 odd-even transposition sort：
    // - 偶數 phase：配對 (0,1), (2,3), ...；
    // - 奇數 phase：配對 (1,2), (3,4), ...。
    // 每次與鄰居交換資料並合併，低 rank 取最小的 local_n 個，高 rank 取最大的 local_n 個
    for(int phase = 0; phase < size+1; phase++){
        int changed_local = 0;
        if(phase % 2 == 0){ // even phase：偶數 phase 配對 (0,1), (2,3), ...
            if(rank % 2 == 0 && rank + 1 < size){
                // 低 rank：與 rank+1 交換，合併後保留較小的 local_n 筆
                long neighbor_n = (rank + 1 < extra) ? (base + 1) : base;
                neighbor_data_vec.resize(neighbor_n);
                nvtxRangePop();nvtxRangePush("Comm");
                //左邊的rank傳最大值跟右邊的rank的最小值交換

                MPI_Sendrecv(local_data_vec.data(), local_n, MPI_FLOAT, rank + 1, 0,
                             neighbor_data_vec.data(), neighbor_n, MPI_FLOAT, rank + 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                nvtxRangePop();nvtxRangePush("CPU");
                
                combined_data.resize(local_n + neighbor_n);
                

                
                merge_two(local_data_vec.data(), local_n,
                                neighbor_data_vec.data(), neighbor_n,
                                combined_data.data());
                
                combined_data.resize(local_n); // 低 rank：只保留最小的 local_n 個
                std::swap(combined_data,local_data_vec);
                
            } else if(rank % 2 == 1){
                // 高 rank：與 rank-1 交換，合併後保留較大的 local_n 筆
                long neighbor_n = (rank - 1 < extra) ? (base + 1) : base;
                neighbor_data_vec.resize(neighbor_n);
                nvtxRangePop();nvtxRangePush("Comm");

                MPI_Sendrecv(local_data_vec.data(), local_n, MPI_FLOAT, rank - 1, 0,
                             neighbor_data_vec.data(), neighbor_n, MPI_FLOAT, rank - 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                nvtxRangePop();nvtxRangePush("CPU");

                combined_data.resize(local_n + neighbor_n);


                merge_two(local_data_vec.data(), local_n,
                                neighbor_data_vec.data(), neighbor_n,
                                combined_data.data());
                
                // 只保留最大的 local_n 個（移除前面多餘的）
                combined_data.erase(combined_data.begin(), combined_data.end() - local_n);
                std::swap(combined_data,local_data_vec);
            }
        } else { // odd phase：奇數 phase 配對 (1,2), (3,4), ...
            if(rank % 2 == 1 && rank + 1 < size){
                // 高 rank：與 rank+1 交換（自己是較低的 index? 這裡在奇數 phase，1 會和 2 互換）
                // 對於奇數 rank，這裡扮演較低 index 的角色，保留較小的 local_n
                long neighbor_n = (rank + 1 < extra) ? (base + 1) : base;
                neighbor_data_vec.resize(neighbor_n);
                nvtxRangePop();nvtxRangePush("Comm");

                MPI_Sendrecv(local_data_vec.data(), local_n, MPI_FLOAT, rank + 1, 1,
                             neighbor_data_vec.data(), neighbor_n, MPI_FLOAT, rank + 1, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                nvtxRangePop();nvtxRangePush("CPU");

                combined_data.resize(local_n + neighbor_n);

                
                merge_two(local_data_vec.data(), local_n,
                                neighbor_data_vec.data(), neighbor_n,
                                combined_data.data());  
                
                combined_data.resize(local_n); // 此處保留較小的 local_n 個
                std::swap(combined_data,local_data_vec);

            } else if(rank % 2 == 0 && rank > 0){
                // 偶數且非 0 的 rank：與 rank-1 交換，合併後保留較大的 local_n
                long neighbor_n = (rank - 1 < extra) ? (base + 1) : base;

                neighbor_data_vec.resize(neighbor_n);

                nvtxRangePop();nvtxRangePush("Comm");
                MPI_Sendrecv(local_data_vec.data(), local_n, MPI_FLOAT, rank - 1, 1,
                             neighbor_data_vec.data(), neighbor_n, MPI_FLOAT, rank - 1, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                nvtxRangePop();nvtxRangePush("CPU");

                combined_data.resize(local_n + neighbor_n);
                
                merge_two(local_data_vec.data(), local_n,
                                neighbor_data_vec.data(), neighbor_n,
                                combined_data.data());
                
                // 保留較大的 local_n 個
                combined_data.erase(combined_data.begin(), combined_data.end() - local_n);
                std::swap(combined_data,local_data_vec);
            }
        }
        // 透過 MPI_Allreduce(MPI_LOR) 檢查是否仍有任一 rank 資料變動，若無則可提前結束
        /*
        int any_changed;
        nvtxRangePush("communication");
        MPI_Allreduce(&changed_local, &any_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        nvtxRangePop();
        if (!any_changed) break;
        */
        
    }
    
    nvtxRangePop();nvtxRangePush("IO");
    // 平行寫回各自區段到輸出檔
    MPI_File_open(MPI_COMM_SELF, outfile, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_write_at(fh, file_offset_bytes, local_data_vec.data(), local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    

    // 結束 MPI
    nvtxRangePop();nvtxRangePush("Comm");
    MPI_Barrier(MPI_COMM_WORLD);
    nvtxRangePop();nvtxRangePush("CPU");
    MPI_Finalize();
    return 0;
    nvtxRangePop();
}