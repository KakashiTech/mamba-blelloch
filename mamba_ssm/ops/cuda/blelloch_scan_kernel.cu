// ============================================================================
// Blelloch Tree Scan CUDA Kernel for SSM Exact Prefix Computation
//
// Unlike Triton (which has no inter-thread sync), this kernel uses:
//   - __shared__ memory for tree nodes
//   - __syncthreads() between tree levels
//   - One thread per (dstate, headdim) element
//   - Parallel O(log N) tree reduction + prefix propagation
//
// Template parameters: N (padded seqlen), dstate, headdim
// Grid: (batch * nheads) blocks
// Block: (dstate * headdim) threads
// Shared memory: 2 * N * dstate * headdim floats (scale + drive)
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// ----------------------------------------------------------------------------
// Main FW kernel template
// ----------------------------------------------------------------------------
template<int N, int dstate, int headdim>
__global__ void blelloch_scan_fwd_kernel(
    const float* __restrict__ s_in,     // (B, L, dstate)
    const float* __restrict__ d_in,     // (B, L, dstate, headdim)
    float* __restrict__ h_out,          // (B, L, dstate, headdim)
    int L,
    int stride_s_b, int stride_s_l, int stride_s_d,
    int stride_d_b, int stride_d_l, int stride_d_d, int stride_d_h,
    int stride_h_b, int stride_h_l, int stride_h_d, int stride_h_h)
{
    extern __shared__ float shared[];
    float* s_smem = shared;                             // [N, dstate]
    float* d_smem = shared + N * dstate;                // [N, dstate, headdim]

    int bid = blockIdx.x;                               // batch*head index
    int tid = threadIdx.x;                              // dstate*headdim index

    // -- Load from global to shared (with identity padding) ----------------
    for (int t = 0; t < L; t++) {
        for (int i = tid; i < dstate * headdim; i += blockDim.x) {
            int d_idx = i / headdim;
            int h_idx = i % headdim;
            s_smem[t * dstate + d_idx] = s_in[bid * stride_s_b + t * stride_s_l + d_idx * stride_s_d];
            d_smem[t * dstate * headdim + d_idx * headdim + h_idx] =
                d_in[bid * stride_d_b + t * stride_d_l + d_idx * stride_d_d + h_idx * stride_d_h];
        }
    }
    // Pad remaining to identity
    for (int t = L; t < N; t++) {
        for (int i = tid; i < dstate * headdim; i += blockDim.x) {
            int d_idx = i / headdim;
            int h_idx = i % headdim;
            s_smem[t * dstate + d_idx] = 1.0f;
            d_smem[t * dstate * headdim + d_idx * headdim + h_idx] = 0.0f;
        }
    }
    __syncthreads();

    // -- Up-sweep (tree reduction) -----------------------------------------
    // Each tree level combines pairs separated by `step/2`
    for (int step = 2; step <= N; step <<= 1) {
        int hstep = step >> 1;
        // Nodes at positions (step-1, 2*step-1, 3*step-1, ...) get updated
        for (int node = step - 1; node < N; node += step) {
            int left = node - hstep;
            for (int i = tid; i < dstate * headdim; i += blockDim.x) {
                int d_idx = i / headdim;
                int h_idx = i % headdim;
                float s_l = s_smem[left * dstate + d_idx];
                float s_r = s_smem[node * dstate + d_idx];
                float d_l = d_smem[left * dstate * headdim + d_idx * headdim + h_idx];
                float d_r = d_smem[node * dstate * headdim + d_idx * headdim + h_idx];
                s_smem[node * dstate + d_idx] = s_l * s_r;
                d_smem[node * dstate * headdim + d_idx * headdim + h_idx] = s_r * d_l + d_r;
            }
        }
        __syncthreads();
    }

    // -- Identity at root ---------------------------------------------------
    if (tid < dstate) {
        s_smem[(N - 1) * dstate + tid] = 1.0f;
    }
    for (int i = tid; i < dstate * headdim; i += blockDim.x) {
        int d_idx = i / headdim;
        int h_idx = i % headdim;
        d_smem[(N - 1) * dstate * headdim + d_idx * headdim + h_idx] = 0.0f;
    }
    __syncthreads();

    // -- Down-sweep (prefix propagation) ------------------------------------
    for (int step_val = N.bit_length() - 2; step_val >= 0; --step_val) {
        int cur_step = 1 << (step_val + 1);
        int hstep = cur_step >> 1;
        for (int node = cur_step - 1; node < N; node += cur_step) {
            int left = node - hstep;
            for (int i = tid; i < dstate * headdim; i += blockDim.x) {
                int d_idx = i / headdim;
                int h_idx = i % headdim;
                float s_par = s_smem[node * dstate + d_idx];
                float s_old = s_smem[left * dstate + d_idx];
                float d_par = d_smem[node * dstate * headdim + d_idx * headdim + h_idx];
                float d_old = d_smem[left * dstate * headdim + d_idx * headdim + h_idx];

                s_smem[left * dstate + d_idx] = s_par;
                d_smem[left * dstate * headdim + d_idx * headdim + h_idx] = d_par;
                s_smem[node * dstate + d_idx] = s_par * s_old;
                d_smem[node * dstate * headdim + d_idx * headdim + h_idx] = s_old * d_par + d_old;
            }
        }
        __syncthreads();
    }

    // -- Write output (exclusive prefix -> positions 1..L) ------------------
    for (int t = 0; t < L; t++) {
        for (int i = tid; i < dstate * headdim; i += blockDim.x) {
            int d_idx = i / headdim;
            int h_idx = i % headdim;
            h_out[bid * stride_h_b + t * stride_h_l + d_idx * stride_h_d + h_idx * stride_h_h] =
                d_smem[(t + 1) * dstate * headdim + d_idx * headdim + h_idx];
        }
    }
}

// ----------------------------------------------------------------------------
// BW kernel: reverse scan for adjoint states
// ----------------------------------------------------------------------------
template<int N, int dstate, int headdim>
__global__ void blelloch_scan_bwd_kernel(
    const float* __restrict__ dout_in,  // (B, L, dstate, headdim)
    const float* __restrict__ s_in,     // (B, L, dstate)
    float* __restrict__ ds_out,        // (B, L, dstate)
    float* __restrict__ dd_out,        // (B, L, dstate, headdim)
    int L,
    int stride_dout_b, int stride_dout_l, int stride_dout_d, int stride_dout_h,
    int stride_s_b, int stride_s_l, int stride_s_d,
    int stride_ds_b, int stride_ds_l, int stride_ds_d,
    int stride_dd_b, int stride_dd_l, int stride_dd_d, int stride_dd_h)
{
    extern __shared__ float shared[];
    float* s_rev = shared;                              // [N, dstate]
    float* gout_rev = shared + N * dstate;              // [N, dstate, headdim]

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    // Load s into reverse order: s_rev[t] = s[L-1-t]
    for (int t = 0; t < L; t++) {
        for (int i = tid; i < dstate * headdim; i += blockDim.x) {
            int d_idx = i / headdim;
            int h_idx = i % headdim;
            float s_val = s_in[bid * stride_s_b + t * stride_s_l + d_idx * stride_s_d];
            s_rev[(L - 1 - t) * dstate + d_idx] = s_val;
        }
    }
    // Reverse scan needs gout_rev[t] = dout[L-1-t] as initial value
    for (int t = 0; t < L; t++) {
        for (int i = tid; i < dstate * headdim; i += blockDim.x) {
            int d_idx = i / headdim;
            int h_idx = i % headdim;
            float dout_val = dout_in[bid * stride_dout_b + t * stride_dout_l +
                                     d_idx * stride_dout_d + h_idx * stride_dout_h];
            gout_rev[(L - 1 - t) * dstate * headdim + d_idx * headdim + h_idx] = dout_val;
        }
    }
    // Pad remaining
    for (int t = L; t < N; t++) {
        for (int i = tid; i < dstate * headdim; i += blockDim.x) {
            int d_idx = i / headdim;
            int h_idx = i % headdim;
            s_rev[t * dstate + d_idx] = 1.0f;
            gout_rev[t * dstate * headdim + d_idx * headdim + h_idx] = 0.0f;
        }
    }
    __syncthreads();

    // Up-sweep on reverse operators s_rev
    for (int step = 2; step <= N; step <<= 1) {
        int hstep = step >> 1;
        for (int node = step - 1; node < N; node += step) {
            int left = node - hstep;
            for (int i = tid; i < dstate * headdim; i += blockDim.x) {
                int d_idx = i / headdim;
                int h_idx = i % headdim;
                float s_l = s_rev[left * dstate + d_idx];
                float s_r = s_rev[node * dstate + d_idx];
                float g_l = gout_rev[left * dstate * headdim + d_idx * headdim + h_idx];
                float g_r = gout_rev[node * dstate * headdim + d_idx * headdim + h_idx];
                // Compose in reverse: g_new = s_l * g_r + g_l
                gout_rev[node * dstate * headdim + d_idx * headdim + h_idx] = s_l * s_r;
                s_rev[node * dstate + d_idx] = s_l * s_r;
                gout_rev[node * dstate * headdim + d_idx * headdim + h_idx] = s_l * g_r + g_l;
            }
        }
        __syncthreads();
    }

    // Root of reverse = identity for suffix
    if (tid < dstate) {
        s_rev[(N - 1) * dstate + tid] = 1.0f;
    }
    for (int i = tid; i < dstate * headdim; i += blockDim.x) {
        int d_idx = i / headdim;
        int h_idx = i % headdim;
        gout_rev[(N - 1) * dstate * headdim + d_idx * headdim + h_idx] = 0.0f;
    }
    __syncthreads();

    // Down-sweep to propagate suffix prefixes
    for (int step_val = N.bit_length() - 2; step_val >= 0; --step_val) {
        int cur_step = 1 << (step_val + 1);
        int hstep = cur_step >> 1;
        for (int node = cur_step - 1; node < N; node += cur_step) {
            int left = node - hstep;
            for (int i = tid; i < dstate * headdim; i += blockDim.x) {
                int d_idx = i / headdim;
                int h_idx = i % headdim;
                float s_par = s_rev[node * dstate + d_idx];
                float s_old = s_rev[left * dstate + d_idx];
                float g_par = gout_rev[node * dstate * headdim + d_idx * headdim + h_idx];
                float g_old = gout_rev[left * dstate * headdim + d_idx * headdim + h_idx];

                s_rev[left * dstate + d_idx] = s_par;
                gout_rev[left * dstate * headdim + d_idx * headdim + h_idx] = g_par;
                s_rev[node * dstate + d_idx] = s_par * s_old;
                gout_rev[node * dstate * headdim + d_idx * headdim + h_idx] = s_old * g_par + g_old;
            }
        }
        __syncthreads();
    }

    // Write output: reverse back to original order
    for (int t = 0; t < L; t++) {
        for (int i = tid; i < dstate * headdim; i += blockDim.x) {
            int d_idx = i / headdim;
            int h_idx = i % headdim;
            int rev_t = L - 1 - t;
            float g_val = gout_rev[(rev_t + 1) * dstate * headdim + d_idx * headdim + h_idx];
            // ds = g * h_prev, dd = g
            // Here we just store g (adjoint state) - caller computes projections
            ds_out[bid * stride_ds_b + t * stride_ds_l + d_idx * stride_ds_d] = g_val;
            dd_out[bid * stride_dd_b + t * stride_dd_l + d_idx * stride_dd_d + h_idx * stride_dd_h] = g_val;
        }
    }
}

// ----------------------------------------------------------------------------
// Host-side wrapper for forward
// ----------------------------------------------------------------------------
void blelloch_scan_fwd_cuda(
    const float* s, const float* d, float* h_out,
    int B, int L, int dstate, int headdim,
    cudaStream_t stream = nullptr)
{
    // Compute padded N
    int N = 1;
    while (N <= L) N <<= 1;

    // Grid: 1 block per (batch * head)
    // We treat B as batch*nheads already flattened
    int grid_dim = B;

    // Block: one thread per (dstate * headdim) element, up to 1024
    int block_dim = min(dstate * headdim, 1024);

    // Shared memory: s[N][dstate] + d[N][dstate][headdim]
    size_t smem = N * dstate * sizeof(float) + N * dstate * headdim * sizeof(float);

    // Dispatch based on template sizes
    // Note: for production, use a JIT or switch on N/dstate/headdim
    #define DISPATCH(N_, DS_, HD_) \
        if (N <= N_ && dstate == DS_ && headdim == HD_) { \
            int N_padded = 1; while (N_padded <= L) N_padded <<= 1; \
            blelloch_scan_fwd_kernel<N_, DS_, HD_><<<grid_dim, block_dim, smem, stream>>>( \
                s, d, h_out, L, \
                B * L, L, 1, \
                B * L * dstate, L * dstate, dstate, 1, \
                B * L * dstate, L * dstate, dstate, 1); \
            return; \
        }

    // Common configurations
    DISPATCH(64, 4, 4);
    DISPATCH(128, 4, 4);
    DISPATCH(256, 4, 4);
    DISPATCH(512, 4, 4);
    DISPATCH(1024, 4, 4);
    DISPATCH(2048, 4, 4);
    DISPATCH(4096, 4, 4);

    DISPATCH(64, 8, 8);
    DISPATCH(128, 8, 8);
    DISPATCH(256, 8, 8);
    DISPATCH(512, 8, 8);
    DISPATCH(1024, 8, 8);
    DISPATCH(2048, 8, 8);

    DISPATCH(64, 16, 16);
    DISPATCH(128, 16, 16);
    DISPATCH(256, 16, 16);
    DISPATCH(512, 16, 16);
    DISPATCH(1024, 16, 16);

    DISPATCH(64, 32, 32);
    DISPATCH(128, 32, 32);
    DISPATCH(256, 32, 32);
    DISPATCH(512, 32, 32);

    DISPATCH(64, 64, 64);
    DISPATCH(128, 64, 64);

    // Fallback: print error
    fprintf(stderr, "[blelloch_scan_fwd_cuda] Unsupported config: "
            "N<=%d dstate=%d headdim=%d\n", N, dstate, headdim);
}
