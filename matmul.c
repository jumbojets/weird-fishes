// clang -O2 -o build/matmul matmul.c && ./build/matmul

// TODO:
// * more cache awareness
// * matrix size is non-multiple of 32/64

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "amx.h"
#include "util.h"

#define BS 1
#define M 1024
#define N 1024
#define K 2048

// M1 Max
// L1(perf core): 192+128KB per core
// L1(eff core): 128+64KB per core

// 128 x 128 block sizes: 128 * 128 * 2 * 3 = 96KB
#define BLOCKSIZE_M 128
#define BLOCKSIZE_N 128
#define BLOCKSIZE_K 128

// assume A is transposed to access columns as rows
_Float16 At[BS*K*M] __attribute__ ((aligned (64)));
_Float16  B[BS*K*N] __attribute__ ((aligned (64)));
_Float16  C[BS*M*N] __attribute__ ((aligned (64)));

void matmul() {
  for (int b = 0; b < BS; b++) {

    for (int bi = 0; bi < M; bi+=BLOCKSIZE_M) {      // cache tiling
      for (int bj = 0; bj < N; bj+=BLOCKSIZE_N) {

        for (int i = 0; i < BLOCKSIZE_M; i+=32) {    // Z regtile row
          for (int j = 0; j < BLOCKSIZE_N; j+=64) {  // Z regtile col. process two at once on seperate z-accumulators
            AMX_SET(); // z-register reset to zero

            for (int k = 0; k < K; k+=32) {          // down cols of At and B

              // the X,Y registers can only hold 8 partial rows of 32 f16s
              for (int rb = 0; rb < 32; rb+=8) {
                #pragma clang loop unroll(full)
                for (uint64_t r = 0; r < 8; r++) {
                  AMX_LDY(At + b*K*M + (k+rb+r)*M + bi+i, r, 0);
                  uint64_t xr1 = (2*r)%8; // TODO: there should be cleaner way of doing this
                  uint64_t xr2 = (2*r+1)%8;
                  AMX_LDX(B + b*K*N + (k+rb+r)*N + bj+j, xr1, 0);
                  AMX_LDX(B + b*K*N + (k+rb+r)*N + bj+j + 32, xr2, 0);
                  AMX_FMA16(r*64, xr1*64, 0, 0);
                  AMX_FMA16(r*64, xr2*64, 1, 0);
                }
              }
            }

            #pragma clang loop unroll_count(8)
            for (uint64_t r = 0; r < 32; r++) {
              AMX_STZ(C + b*M*N + (bi+i+r)*N + bj+j, 2*r, 0);
              AMX_STZ(C + b*M*N + (bi+i+r)*N + bj+j + 32, 2*r+1, 0);
            }

            AMX_CLR();
          }
        }
      }
    }
  }
}

#define ITERATIONS 10
#define CHECK_EQUIV 0
#define EPSILON 1e-5

int main() {
  srand(time(NULL));

  uint64_t start, end;

  for (int i = 0; i < ITERATIONS; i++) {
    rand_array(At, BS*K*M);
    rand_array(B, BS*K*N);
    memset(C, 0, BS*M*N*sizeof(int16_t));

    start = clock_gettime_nsec_np(CLOCK_REALTIME);
    matmul();
    end = clock_gettime_nsec_np(CLOCK_REALTIME);

    double gflop = (2.0*BS*M*N*K)*1e-9;
    double s = (end-start)*1e-9;
    printf("%f GFLOP/s -- %.2f ms\n", gflop/s, s*1e3);

#if CHECK_EQUIV
    for (int b = 0; b < BS; b++) {
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          _Float16 real = 0;
          for (int k = 0; k < K; k++) real += At[b*K*M + k*M + i] * B[b*K*N + k*N + j];
          if ((C[b*M*N + i*N + j] - real) > EPSILON) {
            printf("not equivalent at (%d, %d): %f != %f\n", i, j, (float)C[b*M*N + i*N + j], (float)real);
            return 1;
          }
        }
      }
    }
    printf("equivalence test passed\n");
#endif
  }

  return 0;
}
