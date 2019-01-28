#include "../common/common.hpp"
#include <cassert>
#include <cstdio>

extern "C" void hypterm_gold (double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, int);
extern "C" void host_code (double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double, double, double, int, int, int);

int main(int argc, char** argv) {
  int N = 300;

  double (*cons_1)[N][N] = (double (*)[N][N]) getRandom3DArray<double>(N, N, N);
  double (*cons_2)[N][N] = (double (*)[N][N]) getRandom3DArray<double>(N, N, N);
  double (*cons_3)[N][N] = (double (*)[N][N]) getRandom3DArray<double>(N, N, N);
  double (*cons_4)[N][N] = (double (*)[N][N]) getRandom3DArray<double>(N, N, N);
  double (*q_1)[N][N] = (double (*)[N][N]) getRandom3DArray<double>(N, N, N);
  double (*q_2)[N][N] = (double (*)[N][N]) getRandom3DArray<double>(N, N, N);
  double (*q_3)[N][N] = (double (*)[N][N]) getRandom3DArray<double>(N, N, N);
  double (*q_4)[N][N] = (double (*)[N][N]) getRandom3DArray<double>(N, N, N);
  double (*flux_0)[N][N] = (double (*)[N][N]) getZero3DArray<double>(N, N, N);
  double (*flux_1)[N][N] = (double (*)[N][N]) getZero3DArray<double>(N, N, N);
  double (*flux_2)[N][N] = (double (*)[N][N]) getZero3DArray<double>(N, N, N);
  double (*flux_3)[N][N] = (double (*)[N][N]) getZero3DArray<double>(N, N, N);
  double (*flux_4)[N][N] = (double (*)[N][N]) getZero3DArray<double>(N, N, N);
  double (*flux_gold_0)[N][N] = (double (*)[N][N]) getZero3DArray<double>(N, N, N);
  double (*flux_gold_1)[N][N] = (double (*)[N][N]) getZero3DArray<double>(N, N, N);
  double (*flux_gold_2)[N][N] = (double (*)[N][N]) getZero3DArray<double>(N, N, N);
  double (*flux_gold_3)[N][N] = (double (*)[N][N]) getZero3DArray<double>(N, N, N);
  double (*flux_gold_4)[N][N] = (double (*)[N][N]) getZero3DArray<double>(N, N, N);
  double (*dxinv) = (double*) malloc (sizeof (double) * 3);
  dxinv[0] = 0.01f;
  dxinv[1] = 0.02f;
  dxinv[2] = 0.03f;

  hypterm_gold ((double*)flux_gold_0, (double*)flux_gold_1, (double*)flux_gold_2, (double*)flux_gold_3, (double*)flux_gold_4, (double*)cons_1, (double*)cons_2, (double*)cons_3, (double*)cons_4, (double*)q_1, (double*)q_2, (double*)q_3, (double*)q_4, dxinv, N);
  host_code ((double*)flux_0, (double*)flux_1, (double*)flux_2, (double*)flux_3, (double*)flux_4, (double*)cons_1, (double*)cons_2, (double*)cons_3, (double*)cons_4, (double*)q_1, (double*)q_2, (double*)q_3, (double*)q_4, dxinv[0], dxinv[1], dxinv[2], N, N, N);

  double error_0 = checkError3D<double> (N, N, (double*)flux_0, (double*)flux_gold_0, 4, N-4, 4, N-4, 4, N-4);
  printf("[Test] RMS Error : %e\n",error_0);
  if (error_0 > TOLERANCE)
    return -1;
  double error_1 = checkError3D<double> (N, N, (double*)flux_1, (double*)flux_gold_1, 4, N-4, 4, N-4, 4, N-4);
  printf("[Test] RMS Error : %e\n",error_1);
  if (error_1 > TOLERANCE)
    return -1;
  double error_2 = checkError3D<double> (N, N, (double*)flux_2, (double*)flux_gold_2, 4, N-4, 4, N-4, 4, N-4);
  printf("[Test] RMS Error : %e\n",error_2);
  if (error_2 > TOLERANCE)
    return -1;
  double error_3 = checkError3D<double> (N, N, (double*)flux_3, (double*)flux_gold_3, 4, N-4, 4, N-4, 4, N-4);
  printf("[Test] RMS Error : %e\n",error_3);
  if (error_3 > TOLERANCE)
    return -1;
  double error_4 = checkError3D<double> (N, N, (double*)flux_4, (double*)flux_gold_4, 4, N-4, 4, N-4, 4, N-4);
  printf("[Test] RMS Error : %e\n",error_4);
  if (error_4 > TOLERANCE)
    return -1;

  delete[] cons_1;
  delete[] cons_2;
  delete[] cons_3;
  delete[] cons_4;
  delete[] q_1;
  delete[] q_2;
  delete[] q_3;
  delete[] q_4;
  delete[] flux_0;
  delete[] flux_1;
  delete[] flux_2;
  delete[] flux_3;
  delete[] flux_4;
  delete[] flux_gold_0;
  delete[] flux_gold_1;
  delete[] flux_gold_2;
  delete[] flux_gold_3;
  delete[] flux_gold_4;
}
