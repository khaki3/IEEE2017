#include "../common/common.hpp"


static void poisson_step
(const double* l_input, int height, int width_y, int width_x, double* l_output, int step)
{
  const double (*input)[width_y][width_x] =
    (const double (*)[width_y][width_x])l_input;
  double (*output)[width_y][width_x] = (double (*)[width_y][width_x])l_output;

  for (int i = 1; i < height-1; i++)
    for (int j = 1; j < width_y-1; j++)
      for (int k = 1; k < width_x-1; k++) {
          output[i][j][k] = 2.666f * input[i][j][k] - 
		0.166f * (input[i-1][j][k] + input[i+1][j][k] + input[i][j-1][k] + input[i][j+1][k] + input[i][j][k+1] + input[i][j][k-1]) - 
		0.0833f * (input[i-1][j-1][k] + input[i+1][j-1][k] + input[i-1][j+1][k] + input[i+1][j+1][k] + 
			   input[i-1][j][k-1] + input[i+1][j][k-1] + input[i][j-1][k-1] + input[i][j+1][k-1] + 
			   input[i-1][j][k+1] + input[i+1][j][k+1] + input[i][j-1][k+1] + input[i][j+1][k+1]);
      }
}


extern "C" void poisson_gold
(double *l_input, double *l_output, int height, int width_y, int width_x)
{
  double* temp = getZero3DArray<double>(height, width_y, width_x);
  poisson_step(l_input, height, width_y, width_x, temp, 0);
  poisson_step(temp, height, width_y, width_x, l_output, 1);
  memset(temp, 0, sizeof(double) * height * width_y * width_x);
  poisson_step(l_output, height, width_y, width_x, temp, 2);
  memset(l_output, 0, sizeof(double) * height * width_y * width_x);
  poisson_step(temp, height, width_y, width_x, l_output, 3);
  delete[] temp;
}
