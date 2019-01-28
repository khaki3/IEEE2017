#include "../common/common.hpp"
#include <cassert>
#include <cstdio>

extern "C" void jacobi(double*, int, int, double*);

extern "C" void jacobi_gold(double*, int, int, double*);

int main(int argc, char** argv) {
  int width_y, width_x;
  if (argc == 3) {
    width_y = atoi(argv[1]);
    width_x = atoi(argv[2]);
  }
  else {
    width_y = 8192;
    width_x = 8192;
  }

  double (*input)[width_x] = (double (*)[width_x])
    getRandom2DArray<double>(width_y, width_x);
  double (*output)[width_x] = (double (*)[width_x])
    getZero2DArray<double>(width_y, width_x);
  double (*output_gold)[width_x] = (double (*)[width_x])
    getZero2DArray<double>(width_y, width_x);

  jacobi((double*)input, width_y, width_x, (double*)output);

  jacobi_gold((double*)input, width_y, width_x, (double*)output_gold);

#ifdef PRINT_OUTPUT
  printf("Output :\n");
  print2DArray<double>(width_x, (double*)output, 8, width_y-8, 8, width_x-8);
  printf("\nOutput Gold:\n");
  print2DArray<double>(width_x, (double*)output_gold, 8, width_y-8, 8, width_x-8);
#endif

  double error =
    checkError2D<double>
    (width_x, (double*)output, (double*) output_gold, 8, width_y-8, 8, width_x-8);
  printf("[Test] RMS Error : %e\n",error);
  if (error > TOLERANCE)
    return -1;

  delete[] input;
  delete[] output;
  delete[] output_gold;
}
