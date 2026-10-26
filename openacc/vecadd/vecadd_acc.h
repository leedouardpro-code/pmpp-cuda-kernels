#ifndef VECADD_ACC_H
#define VECADD_ACC_H

// C = A + B on N floats, offloaded to the GPU with OpenACC directives.
// Same computation as ch2/vecadd (CUDA), used to compare a directive-based
// port with the hand-written kernel under the same benchmark protocol.

// Compute only: A, B and C must already be present in GPU memory
// (inside an enclosing `acc data` region). Used by the benchmark, so that
// host-device transfers are excluded from the timing, as for the CUDA kernels.
void vec_add_acc_compute(const float* A, const float* B, float* C, int N);

// Complete operation: copies A and B to the GPU, computes, copies C back.
// Used by the correctness test.
void vec_add_acc(const float* A, const float* B, float* C, int N);

void vec_add_cpu(const float* A, const float* B, float* C, int N);

#endif  // VECADD_ACC_H
