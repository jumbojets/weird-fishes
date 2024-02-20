#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "amx.h"

void rand_array(_Float16 *arr, int size) {
  for (int i = 0; i < size; i++)
    arr[i] = (_Float16)((float)rand() / RAND_MAX - 0.5);
}

void print_mat(_Float16 *arr, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++)
      printf("%f, ", (float)arr[i*cols+j]);
    printf("\n");
  }
}

void read_x(_Float16 ret[256]) {
  for (uint64_t i = 0; i < 8; i++)
    AMX_STX(ret+i*32, i, 0);
}

void read_y(_Float16 ret[256]) {
  for (uint64_t i = 0; i < 8; i++)
    AMX_STY(ret+i*32, i, 0);
}

void read_z(_Float16 ret[2048]) {
  for (uint64_t i = 0; i < 64; i++)
    AMX_STZ(ret+i*32, i, 0);
}
