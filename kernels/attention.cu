// attention.cu — single-sequence causal SDPA (fp32) with GQA + optional sliding window.
// Correctness-first: one block per (query position i, query head h). Threads cooperate to
// compute the dot-product scores, a two-pass softmax in shared memory, then the weighted V sum.
#include "attention.h"
#include <cfloat>

#define ATT_THREADS 256

// grid = (n_heads, seq), block = ATT_THREADS
// dynamic shared layout: [ Qs(head_dim) | scores(seq) ]
__global__ void sdpa_kernel(float* out, const float* Q, const float* K, const float* V,
                            int seq, int n_heads, int n_kv, int head_dim,
                            int sliding_window, float scaling) {
    const int h = blockIdx.x;          // query head
    const int i = blockIdx.y;          // query position
    const int tid = threadIdx.x;
    const int group = n_heads / n_kv;  // query heads per kv head
    const int kv = h / group;          // shared kv head for this query head

    extern __shared__ float smem[];
    float* Qs     = smem;              // head_dim
    float* scores = smem + head_dim;   // seq (only [lo..i] used)
    __shared__ float red[ATT_THREADS];
    __shared__ float m_sh, l_sh;

    // lowest valid key position (causal, plus sliding window if enabled)
    int lo = 0;
    if (sliding_window > 0) {
        lo = i - sliding_window + 1;
        if (lo < 0) lo = 0;
    }

    // load this query vector into shared memory
    const float* q = Q + ((size_t)i * n_heads + h) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) Qs[d] = q[d];
    __syncthreads();

    // pass 0: scores[j] = scaling * dot(Q[i,h,:], K[j,kv,:]) for valid j
    for (int j = lo + tid; j <= i; j += blockDim.x) {
        const float* k = K + ((size_t)j * n_kv + kv) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) dot += Qs[d] * k[d];
        scores[j] = dot * scaling;
    }
    __syncthreads();

    // pass 1a: row max over valid j
    float lmax = -FLT_MAX;
    for (int j = lo + tid; j <= i; j += blockDim.x) lmax = fmaxf(lmax, scores[j]);
    red[tid] = lmax; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    if (tid == 0) m_sh = red[0];
    __syncthreads();
    const float m = m_sh;

    // pass 1b: sum of exp(score - m)
    float lsum = 0.f;
    for (int j = lo + tid; j <= i; j += blockDim.x) lsum += expf(scores[j] - m);
    red[tid] = lsum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) l_sh = red[0];
    __syncthreads();
    const float inv_l = 1.f / l_sh;

    // normalize scores into softmax weights
    for (int j = lo + tid; j <= i; j += blockDim.x) scores[j] = expf(scores[j] - m) * inv_l;
    __syncthreads();

    // pass 2: out[i,h,d] = sum_j weight[j] * V[j,kv,d]  (parallel over d, no atomics)
    float* o = out + ((size_t)i * n_heads + h) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int j = lo; j <= i; ++j) acc += scores[j] * V[((size_t)j * n_kv + kv) * head_dim + d];
        o[d] = acc;
    }
}

void sdpa(float* out, const float* Q, const float* K, const float* V,
          int seq, int n_heads, int n_kv, int head_dim, int sliding_window, float scaling, cudaStream_t s) {
    dim3 grid(n_heads, seq);
    size_t shmem = (size_t)(head_dim + seq) * sizeof(float);
    sdpa_kernel<<<grid, ATT_THREADS, shmem, s>>>(out, Q, K, V, seq, n_heads, n_kv,
                                                 head_dim, sliding_window, scaling);
}
