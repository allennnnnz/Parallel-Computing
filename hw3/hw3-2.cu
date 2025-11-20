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
__global__ void phase1(int* Dist, int nPad, int B, int r)
{
    int ti = threadIdx.y;     // 0..B-1
    int tj = threadIdx.x;     // 0..B-1

    int gi = r * B + ti;
    int gj = r * B + tj;

    __shared__ int pivot[32 * 32];    // 1D shared buffer

    int local_idx = ti * B + tj;

    // load pivot block
    pivot[local_idx] = Dist[gi * nPad + gj];

    __syncthreads();

    // k-loop inside pivot block (local k = 0..B-1)
    for (int k = 0; k < B; k++)
    {
        int w1 = pivot[ti * B + k];  // pivot(i,k)
        int w2 = pivot[k * B + tj];  // pivot(k,j)

        int via = (w1 == INF || w2 == INF) ? INF : (w1 + w2);

        

        if (via < pivot[local_idx])
            pivot[local_idx] = via;

        __syncthreads(); // wait all threads update pivot
    }

    // write back
    Dist[gi * nPad + gj] = pivot[local_idx];
}

// Phase 2: pivot row & pivot column blocks
// gridDim = (numBlocks-1, 2)
//   y=0 → row blocks (r, coord)
//   y=1 → col blocks (coord, r)
__global__ void phase2(
    int* Dist,
    int  nPad,
    int  B,
    int  r,
    int  numBlocks
){
    int ti = threadIdx.y;
    int tj = threadIdx.x;
    int local = ti * B + tj;

    // ---- shared memory: two tiles (rowPart & colPart) ----
    __shared__ int tileA[32*32]; // my block
    __shared__ int tileB[32*32]; // pivot row / pivot col block

    int which = blockIdx.y;  // 0=row, 1=col
    int t = blockIdx.x;
    int coord = (t < r) ? t : t + 1;

    int bi = (which == 0 ? r : coord);
    int bj = (which == 0 ? coord : r);

    int gi = bi * B + ti;
    int gj = bj * B + tj;

    // ---- 1. load my block into shared memory ----
    tileA[local] = Dist[gi * nPad + gj];

    // ---- 2. load needed pivot row/col block into tileB ----
    if (which == 0) {
        // row block: we need pivot row block (r, r)
        tileB[local] = Dist[(r * B + ti) * nPad + (r * B + tj)];
    } else {
        // col block: we need pivot col block (r, r)
        tileB[local] = Dist[(r * B + ti) * nPad + (r * B + tj)];
    }

    __syncthreads();

    // ---- 3. Do k-loop fully in shared memory ----
    for (int k = 0; k < B; k++)
    {
        int w1, w2;

        if (which == 0) {   
            // (r, j) = pivotRow[i][k], selfBlock[k][j]
            w1 = tileB[ti * B + k];
            w2 = tileA[k * B + tj];
        } else {            
            // (i, r) = selfBlock[i][k], pivotCol[k][j]
            w1 = tileA[ti * B + k];
            w2 = tileB[k * B + tj];
        }

        int via = (w1 == INF || w2 == INF ? INF : (w1 + w2));

        if (via < tileA[local])
            tileA[local] = via;

        __syncthreads();
    }

    // ---- 4. write back ----
    Dist[gi * nPad + gj] = tileA[local];
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
    int ti = threadIdx.y;  // 0..B-1
    int tj = threadIdx.x;  // 0..B-1

    int block_i = blockIdx.y;
    int block_j = blockIdx.x;

    // 跳過 pivot row 和 pivot col
    if (block_i == r || block_j == r)
        return;

    // global index of this tile
    int gi = block_i * B + ti;
    int gj = block_j * B + tj;

    // pivot-row tile = (r, block_j)
    int pri = r * B + ti;
    int prj = block_j * B + tj;

    // pivot-col tile = (block_i, r)
    int pci = block_i * B + ti;
    int pcj = r * B + tj;

    // ---------- shared memory ------------
    __shared__ int tile[32 * 32];      // current block
    __shared__ int pivotRow[32 * 32];  // block (r, j)
    __shared__ int pivotCol[32 * 32];  // block (i, r)

    int lid = ti * B + tj;

    // load three tiles
    tile[lid]     = Dist[gi * nPad + gj];
    pivotRow[lid] = Dist[pri * nPad + prj];
    pivotCol[lid] = Dist[pci * nPad + pcj];

    __syncthreads();

    // local k loop (0..B-1)
    for (int k = 0; k < B; k++)
    {
        int w1 = pivotCol[ti * B + k];   // Dist[i][k]
        int w2 = pivotRow[k * B + tj];   // Dist[k][j]

        int via = (w1 == INF || w2 == INF ? INF : w1 + w2);

        if (via < tile[lid])
            tile[lid] = via;

        __syncthreads();
    }

    // write back final result
    Dist[gi * nPad + gj] = tile[lid];
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
    #pragma omp parallel for collapse(2)
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
    #pragma omp parallel for
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
    #pragma omp parallel for
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
        dim3 gridPhase1(1, 1);
        phase1<<<gridPhase1, blockDim>>>(Dist_d, nPad, B, r);
            

        // ===== Phase 2: pivot row & column =====

        dim3 gridPhase2(rounds - 1, 2);   // x = all blocks except r, y = 0(row),1(col)
        dim3 threadsPhase2(B, B);

        phase2<<<gridPhase2, threadsPhase2>>>(Dist_d, nPad, B, r, rounds);


        // ===== Phase 3: remaining blocks =====

        dim3 gridPhase3(rounds, rounds);
        phase3<<<gridPhase3, blockDim>>>(Dist_d, nPad, B, r, rounds);
    }

    cudaMemcpy(Dist, Dist_d,
                           nPad*nPad*sizeof(int),
                           cudaMemcpyDeviceToHost);
    cudaFree(Dist_d);
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