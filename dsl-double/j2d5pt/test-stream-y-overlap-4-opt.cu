#include "cuda.h"
#ifdef _TIMER_
#include "cuda_profiler_api.h"
#endif
#include "stdio.h"

#define FORMA_MAX(a,b) ( (a) > (b) ? (a) : (b) )
#define max(a,b) FORMA_MAX(a,b)
#define FORMA_MIN(a,b) ( (a) < (b) ? (a) : (b) )
#define min(a,b) FORMA_MIN(a,b)
#define FORMA_CEIL(a,b) ( (a) % (b) == 0 ? (a) / (b) : ((a) / (b)) + 1 )
#define mod(x,y) ( (x) & (y-1))

#ifndef FORMA_MAX_BLOCKDIM_0
#define FORMA_MAX_BLOCKDIM_0 1024
#endif
#ifndef FORMA_MAX_BLOCKDIM_1
#define FORMA_MAX_BLOCKDIM_1 1024
#endif
#ifndef FORMA_MAX_BLOCKDIM_2
#define FORMA_MAX_BLOCKDIM_2 1024
#endif
template<typename T>
__global__ void  __kernel_init__(T* input, T value)
{
  int loc = (int)(blockIdx.x)*(int)(blockDim.x)+(int)(threadIdx.x);
  input[loc] = value;
}


template<typename T>
void initialize_array(T* d_input, int size, T value)
{
  dim3 init_grid(FORMA_CEIL(size,FORMA_MAX_BLOCKDIM_0));
  dim3 init_block(FORMA_MAX_BLOCKDIM_0);
  __kernel_init__<<<init_grid,init_block>>>(d_input,value);
}


void Check_CUDA_Error(const char* message);
/* Device code Begin */
__global__ void __kernel___forma_kernel__0__(float * __restrict__ input, int N, int M, int FORMA_BLOCKDIM_X, int FORMA_BLOCKDIM_Y, float * __restrict__ __var_1__){
  __shared__ float tilevar[2][64*16];
  int rowy = FORMA_BLOCKDIM_Y+8;
  //int threadIdx_y = mod((int)threadIdx.y,2);

  int __iter_0__ = (int)(blockIdx.x)*(int)(FORMA_BLOCKDIM_X-8);
  int __iter_3__ = FORMA_MAX(__iter_0__,0) + (int)(threadIdx.x) ;

  for (int __iter_1__ = 0; __iter_1__ <= N-1; __iter_1__ += FORMA_BLOCKDIM_Y) {
    int __iter_2__ = FORMA_MAX(__iter_1__,0) + (int)(threadIdx.y) ;
    if(__iter_2__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-1),(N-1)) & __iter_3__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-1),(M-1))){
      tilevar[0][__iter_3__-__iter_0__+FORMA_BLOCKDIM_X*mod(__iter_2__,rowy)] = input[__iter_3__+M*__iter_2__];
    }
    __syncthreads();
    int __iter_4__ = FORMA_MAX((__iter_1__-1),1) + (int)(threadIdx.y) ;
    int __iter_5__ = FORMA_MAX((__iter_0__+1),1) + (int)(threadIdx.x) ;
    if( __iter_4__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-2),(N-2)) & __iter_5__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-2),(M-2)) ){
        float __temp_2__ = (tilevar[0][__iter_5__-__iter_0__+FORMA_BLOCKDIM_X*mod((__iter_4__-1),rowy)]);
        float __temp_5__ = (tilevar[0][__iter_5__-1-__iter_0__+FORMA_BLOCKDIM_X*mod(__iter_4__,rowy)]);
        float __temp_6__ = (5 * __temp_2__ + 12 * __temp_5__);
        float __temp_9__ = (tilevar[0][__iter_5__-__iter_0__+FORMA_BLOCKDIM_X*mod(__iter_4__,rowy)]);
        float __temp_10__ = (__temp_6__ + 15 * __temp_9__);
        float __temp_13__ = (tilevar[0][__iter_5__+1-__iter_0__+FORMA_BLOCKDIM_X*mod(__iter_4__,rowy)]);
        float __temp_14__ = (__temp_10__ + 12 * __temp_13__);
        float __temp_17__ = (tilevar[0][__iter_5__-__iter_0__+FORMA_BLOCKDIM_X*mod((__iter_4__+1),rowy)]);
        float __temp_18__ = (__temp_14__ + 5 * __temp_17__);
        float __temp_19__ = (__temp_18__ / 118);
        tilevar[1][__iter_5__-__iter_0__+FORMA_BLOCKDIM_X*mod(__iter_4__,rowy)] = __temp_19__;
    }
    __syncthreads();
    int __iter_10__ = FORMA_MAX((__iter_1__-2),1) + (int)(threadIdx.y) ;
    int __iter_11__ = FORMA_MAX((__iter_0__+2),1) + (int)(threadIdx.x) ;
    if( __iter_10__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-3),(N-2)) & __iter_11__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-3),(M-2)) ){
        float __temp_32__ = (tilevar[1][__iter_11__-__iter_0__+FORMA_BLOCKDIM_X*mod((__iter_10__-1),rowy)]);
        float __temp_35__ = (tilevar[1][__iter_11__+(-1)-__iter_0__+FORMA_BLOCKDIM_X*mod(__iter_10__,rowy)]);
        float __temp_36__ = (5 * __temp_32__ + 12 * __temp_35__);
        float __temp_39__ = (tilevar[1][__iter_11__-__iter_0__+FORMA_BLOCKDIM_X*mod(__iter_10__,rowy)]);
        float __temp_40__ = (__temp_36__ + 15 * __temp_39__);
        float __temp_43__ = (tilevar[1][__iter_11__+1-__iter_0__+FORMA_BLOCKDIM_X*mod(__iter_10__,rowy)]);
        float __temp_44__ = (__temp_40__ + 12 * __temp_43__);
        float __temp_47__ = (tilevar[1][__iter_11__-__iter_0__+FORMA_BLOCKDIM_X*mod((__iter_10__+1),rowy)]);
        float __temp_48__ = (__temp_44__ + 5 * __temp_47__);
        float __temp_49__ = (__temp_48__ / 118);
        tilevar[0][__iter_11__-__iter_0__+FORMA_BLOCKDIM_X*mod(__iter_10__,rowy)] = __temp_49__;
    }
    __syncthreads();
    int __iter_16__ = FORMA_MAX((__iter_1__-3),1) + (int)(threadIdx.y) ;
    int __iter_17__ = FORMA_MAX((__iter_0__+3),1) + (int)(threadIdx.x) ;
    if( __iter_16__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-4),(N-2)) & __iter_17__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-4),(M-2)) ){
        float __temp_60__ = (tilevar[0][__iter_17__-__iter_0__+FORMA_BLOCKDIM_X*mod((__iter_16__-1),rowy)]);
        float __temp_61__ = (tilevar[0][__iter_17__+(-1)-__iter_0__+FORMA_BLOCKDIM_X*mod(__iter_16__,rowy)]);
        float __temp_62__ = (5 * __temp_60__ + 12 * __temp_61__);
        float __temp_63__ = (tilevar[0][__iter_17__-__iter_0__+FORMA_BLOCKDIM_X*mod(__iter_16__,rowy)]);
        float __temp_64__ = (__temp_62__ + 15 * __temp_63__);
        float __temp_65__ = (tilevar[0][__iter_17__+1-__iter_0__+FORMA_BLOCKDIM_X*mod(__iter_16__,rowy)]);
        float __temp_66__ = (__temp_64__ + 12 * __temp_65__);
        float __temp_67__ = (tilevar[0][__iter_17__-__iter_0__+FORMA_BLOCKDIM_X*mod((__iter_16__+1),rowy)]);
        float __temp_68__ = (__temp_66__ + 5 * __temp_67__);
        float __temp_69__ = (__temp_68__ / 118);
        tilevar[1][__iter_17__-__iter_0__+FORMA_BLOCKDIM_X*mod(__iter_16__,rowy)] = __temp_69__;
    }
    __syncthreads();
    int __iter_22__ = FORMA_MAX((__iter_1__-4),1) + (int)(threadIdx.y) ;
    int __iter_23__ = FORMA_MAX((__iter_0__+4),1) + (int)(threadIdx.x) ;
    if( __iter_22__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-5),(N-2)) & __iter_23__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-5),(M-2)) ){
        float __temp_80__ = (tilevar[1][__iter_23__-__iter_0__+FORMA_BLOCKDIM_X*mod((__iter_22__-1),rowy)]);
        float __temp_81__ = (tilevar[1][__iter_23__+(-1)-__iter_0__+FORMA_BLOCKDIM_X*mod(__iter_22__,rowy)]);
        float __temp_82__ = (5 * __temp_80__ + 12 * __temp_81__);
        float __temp_83__ = (tilevar[1][__iter_23__-__iter_0__+FORMA_BLOCKDIM_X*mod(__iter_22__,rowy)]);
        float __temp_84__ = (__temp_82__ + 15 * __temp_83__);
        float __temp_85__ = (tilevar[1][__iter_23__+1-__iter_0__+FORMA_BLOCKDIM_X*mod(__iter_22__,rowy)]);
        float __temp_86__ = (__temp_84__ + 12 * __temp_85__);
        float __temp_87__ = (tilevar[1][__iter_23__-__iter_0__+FORMA_BLOCKDIM_X*mod((__iter_22__+1),rowy)]);
        float __temp_88__ = (__temp_86__ + 5 * __temp_87__);
        float __temp_89__ = (__temp_88__ / 118);
        __var_1__[__iter_23__+(M)*(__iter_22__)] = __temp_89__;
    }
  }
}

/*Device code End */
/* Host Code Begin */
extern "C" void jacobi(float * h_input, int N, int M, float * __var_0__){

/* Host allocation Begin */
  float * input;
  cudaMalloc(&input,sizeof(float)*((N)*(M)));
  Check_CUDA_Error("Allocation Error!! : input\n");
  cudaPointerAttributes ptrAttrib_h_input;
  cudaMemcpyKind memcpy_kind_h_input = cudaMemcpyHostToDevice;
  if (cudaPointerGetAttributes(&ptrAttrib_h_input, h_input) == cudaSuccess)
    if (ptrAttrib_h_input.memoryType == cudaMemoryTypeDevice)
      memcpy_kind_h_input = cudaMemcpyDeviceToDevice;
  cudaGetLastError();
  if( memcpy_kind_h_input != cudaMemcpyDeviceToDevice ){
    cudaMemcpy(input,h_input,sizeof(float)*((N)*(M)), memcpy_kind_h_input);
  }
  float * __var_1__;
  cudaMalloc(&__var_1__,sizeof(float)*((N)*(M)));
  Check_CUDA_Error("Allocation Error!! : __var_1__\n");
/*Host Allocation End */
/* Kernel Launch Begin */
  int __FORMA_MAX_SHARED_MEM__;
  cudaDeviceGetAttribute(&__FORMA_MAX_SHARED_MEM__,cudaDevAttrMaxSharedMemoryPerBlock,0);
#ifdef _TIMER_
  cudaEvent_t _forma_timer_start_,_forma_timer_stop_;
  cudaEventCreate(&_forma_timer_start_);
  cudaEventCreate(&_forma_timer_stop_);
  cudaEventRecord(_forma_timer_start_,0);
#endif
  int __size_0___kernel___forma_kernel__0__ = ((M-1) - 0 ) + 1;
  int __block_0___kernel___forma_kernel__0__ = 64;
  int __block_1___kernel___forma_kernel__0__ = 8;
  dim3 __blockConfig___kernel___forma_kernel__0__(__block_0___kernel___forma_kernel__0__,__block_1___kernel___forma_kernel__0__);
  int __grid_0___kernel___forma_kernel__0__ = FORMA_CEIL(__size_0___kernel___forma_kernel__0__,__blockConfig___kernel___forma_kernel__0__.x-8);
  int __grid_1___kernel___forma_kernel__0__ = 1;
  dim3 __gridConfig___kernel___forma_kernel__0__(__grid_0___kernel___forma_kernel__0__,__grid_1___kernel___forma_kernel__0__);
  __kernel___forma_kernel__0__<<<__gridConfig___kernel___forma_kernel__0__, __blockConfig___kernel___forma_kernel__0__>>> (input, N, M, __blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y, __var_1__);
  Check_CUDA_Error("Kernel Launch Error!! : __kernel___forma_kernel__0__\n");

  cudaPointerAttributes ptrAttrib___var_0__;
  cudaMemcpyKind memcpy_kind___var_0__ = cudaMemcpyDeviceToHost;
  if (cudaPointerGetAttributes(&ptrAttrib___var_0__, __var_0__) == cudaSuccess)
    if (ptrAttrib___var_0__.memoryType == cudaMemoryTypeDevice)
      memcpy_kind___var_0__ = cudaMemcpyDeviceToDevice;
  cudaGetLastError();
  cudaMemcpy(__var_0__,__var_1__, sizeof(float)*((N)*(M)), memcpy_kind___var_0__);
#ifdef _TIMER_
  cudaEventRecord(_forma_timer_stop_,0);
  cudaEventSynchronize(_forma_timer_stop_);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime,_forma_timer_start_,_forma_timer_stop_);
  printf("[FORMA] Computation Time(ms) : %lf\n",elapsedTime);
  cudaEventDestroy(_forma_timer_start_);
  cudaEventDestroy(_forma_timer_stop_);
#endif
/*Kernel Launch End */
/* Host Free Begin */
  cudaFree(input);
  cudaFree(__var_1__);
}
/*Host Free End*/
