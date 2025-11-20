#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define INF 1073741823  // (1<<30)-1

// 簡單的 CUDA error check macro
#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA Error %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(1);                                                \
    }                                                           \
} while (0)


// ======================================================
// Phase Kernels
// ======================================================

// Phase 1: pivot block (r, r)
// 這裡用「單一 thread」在 GPU 上順序跑完 B×B 的 FW，確保正確性
__global__ void phase1(
    int* Dist,
    int  nPad,
    int  B,
    int  r
){
    if (threadIdx.x != 0 || threadIdx.y != 0 ||
        blockIdx.x  != 0 || blockIdx.y  != 0) return;

    int i0 = r * B;
    int j0 = r * B;
    int k_start = r * B;
    int k_end   = k_start + B;  // nPad 已是 B 的倍數，不會越界

    for (int k = k_start; k < k_end; ++k) {
        for (int i = i0; i < i0 + B; ++i) {
            for (int j = j0; j < j0 + B; ++j) {

                int idx_ik = i * nPad + k;
                int idx_kj = k * nPad + j;
                int idx_ij = i * nPad + j;

                int w1 = Dist[idx_ik];
                int w2 = Dist[idx_kj];
                if (w1 == INF || w2 == INF) continue;
                int via = w1 + w2;
                if (via < Dist[idx_ij])
                    Dist[idx_ij] = via;
            }
        }
    }
}

// Phase 2: pivot row & pivot column blocks
// gridDim = (numBlocks-1, 2)
//   y=0 → row blocks (r, coord)
//   y=1 → col blocks (coord, r)
__global__ void phase2(
    int* Dist,
    int  nPad,
    int  B,
    int  r,           // current round index
    int  numBlocks    // nPad / B
){
    int which = blockIdx.y;   // 0: row (r, j)  1: col (i, r)

    // blockIdx.x ∈ [0, numBlocks-2]，映射成跳過 r 的 block index
    int t = blockIdx.x;
    int coord = (t < r) ? t : t + 1;   // coord ∈ [0, numBlocks-1], coord != r

    int block_i, block_j;
    if (which == 0) {
        // pivot row: (r, coord)
        block_i = r;
        block_j = coord;
    } else {
        // pivot column: (coord, r)
        block_i = coord;
        block_j = r;
    }

    int block_start_x = block_i * B;
    int block_start_y = block_j * B;

    int i = block_start_x + threadIdx.y;
    int j = block_start_y + threadIdx.x;

    int k_start = r * B;
    int k_end   = k_start + B;

    for (int k = k_start; k < k_end; ++k) {
        int idx_ik = i * nPad + k;
        int idx_kj = k * nPad + j;
        int idx_ij = i * nPad + j;

        int w1 = Dist[idx_ik];
        int w2 = Dist[idx_kj];
        if (w1 == INF || w2 == INF) continue;
        int via = w1 + w2;
        if (via < Dist[idx_ij])
            Dist[idx_ij] = via;
    }
}

// Phase 3: other blocks (neither in row r nor col r)
// gridDim = (numBlocks, numBlocks)
__global__ void phase3(
    int* Dist,
    int  nPad,
    int  B,
    int  r,
    int  numBlocks
){
    int block_i = blockIdx.y;
    int block_j = blockIdx.x;

    // 跳過 pivot row 和 pivot col
    if (block_i == r || block_j == r)
        return;

    int block_start_x = block_i * B;
    int block_start_y = block_j * B;

    int i = block_start_x + threadIdx.y;
    int j = block_start_y + threadIdx.x;

    int k_start = r * B;
    int k_end   = k_start + B;

    for (int k = k_start; k < k_end; ++k) {
        int idx_ik = i * nPad + k;
        int idx_kj = k * nPad + j;
        int idx_ij = i * nPad + j;

        int w1 = Dist[idx_ik];
        int w2 = Dist[idx_kj];
        if (w1 == INF || w2 == INF) continue;
        int via = w1 + w2;
        if (via < Dist[idx_ij])
            Dist[idx_ij] = via;
    }
}


// ======================================================
// Input / Output（1D 展開）
// ======================================================
void input(char* infile, int B, int** Dist_ptr, int* n_ptr, int* nPad_ptr){
    FILE* file = fopen(infile, "rb");
    if (!file) {
        perror("fopen input");
        exit(1);
    }

    int n, m;
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    *n_ptr = n;

    int nPad = ((n + B - 1) / B) * B;
    *nPad_ptr = nPad;

    int* Dist = (int*)malloc(nPad * nPad * sizeof(int));
    if (!Dist) {
        fprintf(stderr, "malloc Dist failed\n");
        exit(1);
    }

    // 初始化：對角線 0，其餘 INF，padding 區也 INF
    for (int i = 0; i < nPad; ++i) {
        for (int j = 0; j < nPad; ++j) {
            if (i < n && j < n) {
                Dist[i*nPad + j] = (i == j ? 0 : INF);
            } else {
                Dist[i*nPad + j] = INF;
            }
        }
    }

    // 讀邊
    int edge[3];
    for (int i = 0; i < m; ++i) {
        fread(edge, sizeof(int), 3, file);
        int u = edge[0], v = edge[1], w = edge[2];
        Dist[u*nPad + v] = w;
    }

    fclose(file);
    *Dist_ptr = Dist;
}

void output(char* outfile, int* Dist, int n, int nPad){
    FILE* out = fopen(outfile, "wb");
    if (!out) {
        perror("fopen output");
        exit(1);
    }
    for (int i = 0; i < n; ++i) {
        fwrite(&Dist[i*nPad], sizeof(int), n, out);
    }
    fclose(out);
}


// ======================================================
// Host: Blocked FW kernel launcher
// ======================================================
void block_FW_CUDA(int* Dist, int n, int nPad, int B){
    int rounds    = nPad / B; // 一邊有多少個 blocks
    int numBlocks = rounds;

    if (B * B > 1024) {
        fprintf(stderr, "Error: B too large (B*B > 1024)\n");
        exit(1);
    }

    int *Dist_d = NULL;
    CUDA_CHECK( cudaMalloc(&Dist_d, nPad * nPad * sizeof(int)) );
    CUDA_CHECK( cudaMemcpy(Dist_d, Dist,
                           nPad*nPad*sizeof(int),
                           cudaMemcpyHostToDevice) );

    dim3 blockDim(B, B);

    for (int r = 0; r < rounds; ++r) {
        // ===== Phase 1: pivot block (r,r) =====
        {
            dim3 gridPhase1(1, 1);
            phase1<<<gridPhase1, dim3(1,1)>>>(Dist_d, nPad, B, r);
            CUDA_CHECK( cudaDeviceSynchronize() );
        }

        // ===== Phase 2: pivot row & column =====
        {
            dim3 gridPhase2(rounds - 1, 2); // x: 0..rounds-2, y: 0(row)/1(col)
            phase2<<<gridPhase2, blockDim>>>(Dist_d, nPad, B, r, rounds);
            CUDA_CHECK( cudaDeviceSynchronize() );
        }

        // ===== Phase 3: remaining blocks =====
        {
            dim3 gridPhase3(rounds, rounds);
            phase3<<<gridPhase3, blockDim>>>(Dist_d, nPad, B, r, rounds);
            CUDA_CHECK( cudaDeviceSynchronize() );
        }
    }

    CUDA_CHECK( cudaMemcpy(Dist, Dist_d,
                           nPad*nPad*sizeof(int),
                           cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaFree(Dist_d) );
}


// ======================================================
// Main
// ======================================================
int main(int argc, char** argv){
    if (argc < 3) {
        printf("Usage: %s input.bin output.bin\n", argv[0]);
        return 0;
    }

    int *Dist;
    int n, nPad;
    int B = 32;  // 記得確保 B*B <= 1024，B 是 block 大小

    input(argv[1], B, &Dist, &n, &nPad);
    block_FW_CUDA(Dist, n, nPad, B);
    output(argv[2], Dist, n, nPad);

    free(Dist);
    return 0;
}