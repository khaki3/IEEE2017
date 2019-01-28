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

#ifndef FORMA_MAX_BLOCKDIM_0
#define FORMA_MAX_BLOCKDIM_0 1024
#endif
#ifndef FORMA_MAX_BLOCKDIM_1
#define FORMA_MAX_BLOCKDIM_1 1024
#endif
#ifndef FORMA_MAX_BLOCKDIM_2
#define FORMA_MAX_BLOCKDIM_2 1024
#endif
#define GAPX (22)
#define EXTENT (10)

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
/*Texture references */
/*Shared Memory Variable */
extern __shared__ char __FORMA_SHARED_MEM__[];
/* Device code Begin */
__global__ void __kernel___forma_kernel__0__(float * __restrict__ input, int L, int M, int N, float * __restrict__ __copy_arr_0__, float * __restrict__ __copy_arr_1__, float * __restrict__ __copy_arr_2__, int FORMA_BLOCKDIM_X, int FORMA_BLOCKDIM_Y, int FORMA_BLOCKDIM_Z, float * __restrict__ __var_1__){
  int __FORMA_SHARED_MEM_OFFSET__ = 0;
  float* __tilevar_0__ = (float*)(__FORMA_SHARED_MEM__+__FORMA_SHARED_MEM_OFFSET__);
  __FORMA_SHARED_MEM_OFFSET__ += sizeof(float)*(((FORMA_BLOCKDIM_Z-0)*(FORMA_BLOCKDIM_Y-0)*(FORMA_BLOCKDIM_X-0)));
  float* __tilevar_1__ = (float*)(__FORMA_SHARED_MEM__+__FORMA_SHARED_MEM_OFFSET__);
  __FORMA_SHARED_MEM_OFFSET__ += sizeof(float)*(((FORMA_BLOCKDIM_Z-0)*(FORMA_BLOCKDIM_Y-0)*(FORMA_BLOCKDIM_X-0)));
  float * __tilevar_2__ = __tilevar_0__;
  float * __tilevar_3__ = __tilevar_1__;
  float * __tilevar_4__ = __tilevar_0__;
  float * __tilevar_5__ = __tilevar_1__;
  int __iter_0__ = (int)(blockIdx.x)*((int)(FORMA_BLOCKDIM_X)+GAPX);
  int __iter_1__ = (int)(blockIdx.y)*((int)(FORMA_BLOCKDIM_Y));
  int __iter_2__ = (int)(blockIdx.z)*((int)(FORMA_BLOCKDIM_Z));

  int __iter_3__ = FORMA_MAX(__iter_2__,0) + (int)(threadIdx.z) ;
  for( ; __iter_3__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-1),(L-1)) ; __iter_3__+=(int)(blockDim.z) ){
    int __iter_4__ = FORMA_MAX(__iter_1__,0) + (int)(threadIdx.y) ;
    if(__iter_4__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-1),(M-1))) {
      int __iter_5__ = FORMA_MAX(__iter_0__,0) + (int)(threadIdx.x) ;
      if( __iter_5__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-1),(N-1)) ){
        __tilevar_2__[__iter_5__+(0-__iter_0__)+(FORMA_BLOCKDIM_X-0)*(__iter_4__+(0-__iter_1__)+(FORMA_BLOCKDIM_Y-0)*(__iter_3__+(0-__iter_2__)))] = input[__iter_5__+(N-0)*(__iter_4__+(M-0)*(__iter_3__))];
      }
    }
  }
  __syncthreads();
  int __iter_6__ = FORMA_MAX((__iter_2__+1),1) + (int)(threadIdx.z) ;
  for( ; __iter_6__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-2),(L-2)) ; __iter_6__+=(int)(blockDim.z) ){
    int __iter_7__ = FORMA_MAX((__iter_1__+1),1) + (int)(threadIdx.y) ;
    if(__iter_7__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-2),(M-2))) {
      int __iter_8__ = FORMA_MAX((__iter_0__+1),1) + (int)(threadIdx.x) ;
      if( __iter_8__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-2),(N-2)) ){
        float __temp_3__ = (__tilevar_2__[__iter_8__+(1)+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_7__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_6__+(0-(__iter_2__+0))))]);
        float __temp_7__ = (__tilevar_2__[__iter_8__+(-1)+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_7__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_6__+(0-(__iter_2__+0))))]);
        float __temp_8__ = (0.161000f * __temp_3__ + 0.162000f * __temp_7__);
        float __temp_12__ = (__tilevar_2__[__iter_8__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_7__+(1)+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_6__+(0-(__iter_2__+0))))]);
        float __temp_13__ = (__temp_8__ + 0.163000f * __temp_12__);
        float __temp_17__ = (__tilevar_2__[__iter_8__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_7__+(-1)+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_6__+(0-(__iter_2__+0))))]);
        float __temp_18__ = (__temp_13__ + 0.164000f * __temp_17__);
        float __temp_22__ = (__tilevar_2__[__iter_8__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_7__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_6__+(1)+(0-(__iter_2__+0))))]);
        float __temp_23__ = (__temp_18__ + 0.165000f * __temp_22__);
        float __temp_27__ = (__tilevar_2__[__iter_8__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_7__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_6__+(-1)+(0-(__iter_2__+0))))]);
        float __temp_28__ = (__temp_23__ + 0.166000f * __temp_27__);
        float __temp_32__ = (__tilevar_2__[__iter_8__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_7__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*( __iter_6__+(0-(__iter_2__+0))))]);
        float __temp_33__ = (__temp_28__ - 1.670000f * __temp_32__);
        __tilevar_3__[__iter_8__+(0-(__iter_0__+1))+(FORMA_BLOCKDIM_X-0)*(__iter_7__+(0-(__iter_1__+1))+(FORMA_BLOCKDIM_Y-0)*(__iter_6__+(0-(__iter_2__+1))))] = __temp_33__;
      }
    }
  }
  __syncthreads ();
  int __iter_9__ = FORMA_MAX((__iter_2__+1),1) + (int)(threadIdx.z) ;
  for( ; __iter_9__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-2),(L-2)) ; __iter_9__ += (int)(blockDim.z) ){
    int __iter_10__ = FORMA_MAX((__iter_1__+1),1) + (int)(threadIdx.y) ;
    if(__iter_10__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-2),(M-2))) {
      int __iter_11__ = FORMA_MAX((__iter_0__+1),1) + (int)(threadIdx.x) ;
      if( __iter_11__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-2),(N-2)) ){
        if (__iter_9__ < (FORMA_MAX((__iter_2__+1),1)+2) || __iter_9__ > (FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-2),(L-2))-2) || __iter_10__ < (FORMA_MAX((__iter_1__+1),1)+2) || __iter_10__ > (FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-2),(M-2))-2) || __iter_11__ < (FORMA_MAX((__iter_0__+1),1)+2) || __iter_11__ > (FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-2),(N-2))-2)) {
          __copy_arr_0__[__iter_11__+(N-0)*(__iter_10__+(M-0)*(__iter_9__))] = __tilevar_3__[__iter_11__+(0-(__iter_0__+1))+(FORMA_BLOCKDIM_X-0)*(__iter_10__+(0-(__iter_1__+1))+(FORMA_BLOCKDIM_Y-0)*(__iter_9__+(0-(__iter_2__+1))))];
        }
      }
    }
  }
  __syncthreads();
  int __iter_15__ = FORMA_MAX((__iter_2__+2),1) + (int)(threadIdx.z) ;
  for( ; __iter_15__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-3),(L-2)) ; __iter_15__ += (int)(blockDim.z) ){
    int __iter_16__ = FORMA_MAX((__iter_1__+2),1) + (int)(threadIdx.y) ;
    if(__iter_16__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-3),(M-2))) {
      int __iter_17__ = FORMA_MAX((__iter_0__+2),1) + (int)(threadIdx.x) ;
      if( __iter_17__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-3),(N-2)) ){
        float __temp_50__ = (__tilevar_3__[__iter_17__+(1)+(0-(__iter_0__+1))+(FORMA_BLOCKDIM_X-0)*(__iter_16__+(0-(__iter_1__+1))+(FORMA_BLOCKDIM_Y-0)*(__iter_15__+(0-(__iter_2__+1))))]);
        float __temp_54__ = (__tilevar_3__[__iter_17__+(-1)+(0-(__iter_0__+1))+(FORMA_BLOCKDIM_X-0)*(__iter_16__+(0-(__iter_1__+1))+(FORMA_BLOCKDIM_Y-0)*(__iter_15__+(0-(__iter_2__+1))))]);
        float __temp_55__ = (0.161000f * __temp_50__ + 0.162000f * __temp_54__);
        float __temp_59__ = (__tilevar_3__[__iter_17__+(0-(__iter_0__+1))+(FORMA_BLOCKDIM_X-0)*(__iter_16__+(1)+(0-(__iter_1__+1))+(FORMA_BLOCKDIM_Y-0)*(__iter_15__+(0-(__iter_2__+1))))]);
        float __temp_60__ = (__temp_55__ + 0.163000f * __temp_59__);
        float __temp_64__ = (__tilevar_3__[__iter_17__+(0-(__iter_0__+1))+(FORMA_BLOCKDIM_X-0)*(__iter_16__+(-1)+(0-(__iter_1__+1))+(FORMA_BLOCKDIM_Y-0)*(__iter_15__+(0-(__iter_2__+1))))]);
        float __temp_65__ = (__temp_60__ + 0.164000f * __temp_64__);
        float __temp_69__ = (__tilevar_3__[__iter_17__+(0-(__iter_0__+1))+(FORMA_BLOCKDIM_X-0)*(__iter_16__+(0-(__iter_1__+1))+(FORMA_BLOCKDIM_Y-0)*(__iter_15__+(1)+(0-(__iter_2__+1))))]);
        float __temp_70__ = (__temp_65__ + 0.165000f * __temp_69__);
        float __temp_74__ = (__tilevar_3__[__iter_17__+(0-(__iter_0__+1))+(FORMA_BLOCKDIM_X-0)*(__iter_16__+(0-(__iter_1__+1))+(FORMA_BLOCKDIM_Y-0)*(__iter_15__+(-1)+(0-(__iter_2__+1))))]);
        float __temp_75__ = (__temp_70__ + 0.166000f * __temp_74__);
        float __temp_79__ = (__tilevar_3__[__iter_17__+(0-(__iter_0__+1))+(FORMA_BLOCKDIM_X-0)*(__iter_16__+(0-(__iter_1__+1))+(FORMA_BLOCKDIM_Y-0)*(__iter_15__+(0-(__iter_2__+1))))]);
        float __temp_80__ = (__temp_75__ - 1.670000f * __temp_79__);
        __tilevar_4__[__iter_17__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__iter_16__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__iter_15__+(0-(__iter_2__+2))))] = __temp_80__;
      }
    }
  }
  __syncthreads ();
  int __iter_18__ = FORMA_MAX((__iter_2__+2),1) + (int)(threadIdx.z) ;
  for( ; __iter_18__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-3),(L-2)) ; __iter_18__ += (int)(blockDim.z) ){
    int __iter_19__ = FORMA_MAX((__iter_1__+2),1) + (int)(threadIdx.y) ;
    if(__iter_19__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-3),(M-2))) {
      int __iter_20__ = FORMA_MAX((__iter_0__+2),1) + (int)(threadIdx.x) ;
      if( __iter_20__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-3),(N-2)) ){
        if (__iter_18__ < (FORMA_MAX((__iter_2__+2),1)+2) || __iter_18__ > (FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-3),(L-2))-2) || __iter_19__ < (FORMA_MAX((__iter_1__+2),1)+2) || __iter_19__ > (FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-3),(M-2))-2) || __iter_20__ < (FORMA_MAX((__iter_0__+2),1)+2) || __iter_20__ > (FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-3),(N-2))-2)) {
          __copy_arr_1__[__iter_20__+(N-0)*(__iter_19__+(M-0)*(__iter_18__))] = __tilevar_4__[__iter_20__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__iter_19__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__iter_18__+(0-(__iter_2__+2))))];
        }
      }
    }
  }
  __syncthreads();
  int __iter_24__ = FORMA_MAX((__iter_2__+3),1) + (int)(threadIdx.z) ;
  for( ; __iter_24__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-4),(L-2)) ; __iter_24__ += (int)(blockDim.z) ){
    int __iter_25__ = FORMA_MAX((__iter_1__+3),1) + (int)(threadIdx.y) ;
    if(__iter_25__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-4),(M-2))) {
      int __iter_26__ = FORMA_MAX((__iter_0__+3),1) + (int)(threadIdx.x) ;
      if( __iter_26__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-4),(N-2)) ){
        float __temp_94__ = (__tilevar_4__[__iter_26__+(1)+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+2))))]);
        float __temp_95__ = (__tilevar_4__[__iter_26__+(-1)+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+2))))]);
        float __temp_96__ = (0.161000f * __temp_94__ + 0.162000f * __temp_95__);
        float __temp_97__ = (__tilevar_4__[__iter_26__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(1)+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+2))))]);
        float __temp_98__ = (__temp_96__ + 0.163000f * __temp_97__);
        float __temp_99__ = (__tilevar_4__[__iter_26__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(-1)+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+2))))]);
        float __temp_100__ = (__temp_98__ + 0.164000f * __temp_99__);
        float __temp_101__ = (__tilevar_4__[__iter_26__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(1)+(0-(__iter_2__+2))))]);
        float __temp_102__ = (__temp_100__ + 0.165000f * __temp_101__);
        float __temp_103__ = (__tilevar_4__[__iter_26__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(-1)+(0-(__iter_2__+2))))]);
        float __temp_104__ = (__temp_102__ + 0.166000f * __temp_103__);
        float __temp_105__ = (__tilevar_4__[__iter_26__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+2))))]);
        float __temp_106__ = (__temp_104__ - 1.670000f * __temp_105__);
        __tilevar_5__[__iter_26__+(0-(__iter_0__+3))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+3))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+3))))] = __temp_106__;
      }
    }
  }
  __syncthreads ();
  int __iter_27__ = FORMA_MAX((__iter_2__+3),1) + (int)(threadIdx.z) ;
  for( ; __iter_27__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-4),(L-2)) ; __iter_27__ += (int)(blockDim.z) ){
    int __iter_28__ = FORMA_MAX((__iter_1__+3),1) + (int)(threadIdx.y) ;
    if(__iter_28__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-4),(M-2))) {
      int __iter_29__ = FORMA_MAX((__iter_0__+3),1) + (int)(threadIdx.x) ;
      if( __iter_29__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-4),(N-2)) ){
        if (__iter_27__ < (FORMA_MAX((__iter_2__+3),1)+2) || __iter_27__ > (FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-4),(L-2))-2) || __iter_28__ < (FORMA_MAX((__iter_1__+3),1)+2) || __iter_28__ > (FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-4),(M-2))-2) || __iter_29__ < (FORMA_MAX((__iter_0__+3),1)+2) || __iter_29__ > (FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-4),(N-2))-2)) {
          __copy_arr_2__[__iter_29__+(N-0)*(__iter_28__+(M-0)*(__iter_27__))] = __tilevar_5__[__iter_29__+(0-(__iter_0__+3))+(FORMA_BLOCKDIM_X-0)*(__iter_28__+(0-(__iter_1__+3))+(FORMA_BLOCKDIM_Y-0)*(__iter_27__+(0-(__iter_2__+3))))];
        }
      }
    }
  }
  __syncthreads();
  int __iter_33__ = FORMA_MAX((__iter_2__+4),1) + (int)(threadIdx.z) ;
  for( ; __iter_33__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-5),(L-2)) ; __iter_33__ += (int)(blockDim.z) ){
    int __iter_34__ = FORMA_MAX((__iter_1__+4),1) + (int)(threadIdx.y) ;
    if(__iter_34__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-5),(M-2))) {
      int __iter_35__ = FORMA_MAX((__iter_0__+4),1) + (int)(threadIdx.x) ;
      if( __iter_35__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-5),(N-2)) ){
        float __temp_120__ = (__tilevar_5__[__iter_35__+(1)+(0-(__iter_0__+3))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+3))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+3))))]);
        float __temp_121__ = (__tilevar_5__[__iter_35__+(-1)+(0-(__iter_0__+3))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+3))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+3))))]);
        float __temp_122__ = (0.161000f * __temp_120__ + 0.162000f * __temp_121__);
        float __temp_123__ = (__tilevar_5__[__iter_35__+(0-(__iter_0__+3))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(1)+(0-(__iter_1__+3))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+3))))]);
        float __temp_124__ = (__temp_122__ + 0.163000f * __temp_123__);
        float __temp_125__ = (__tilevar_5__[__iter_35__+(0-(__iter_0__+3))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(-1)+(0-(__iter_1__+3))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+3))))]);
        float __temp_126__ = (__temp_124__ + 0.164000f * __temp_125__);
        float __temp_127__ = (__tilevar_5__[__iter_35__+(0-(__iter_0__+3))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+3))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(1)+(0-(__iter_2__+3))))]);
        float __temp_128__ = (__temp_126__ + 0.165000f * __temp_127__);
        float __temp_129__ = (__tilevar_5__[__iter_35__+(0-(__iter_0__+3))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+3))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(-1)+(0-(__iter_2__+3))))]);
        float __temp_130__ = (__temp_128__ + 0.166000f * __temp_129__);
        float __temp_131__ = (__tilevar_5__[__iter_35__+(0-(__iter_0__+3))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+3))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+3))))]);
        float __temp_132__ = (__temp_130__ - 1.670000f * __temp_131__);
        __var_1__[__iter_35__+(N-0)*(__iter_34__+(M-0)*(__iter_33__))] = __temp_132__;
      }
    }
  }
}

int __blockSizeToSMemSize___kernel___forma_kernel__0__(dim3 blockDim){
  int FORMA_BLOCKDIM_Z = (int)(blockDim.z);
  int FORMA_BLOCKDIM_Y = (int)(blockDim.y);
  int FORMA_BLOCKDIM_X = (int)(blockDim.x);
  int SMemSize = 0;
  SMemSize += sizeof(float)*(((FORMA_BLOCKDIM_Z-0)*(FORMA_BLOCKDIM_Y-0)*(FORMA_BLOCKDIM_X-0)));
  SMemSize += sizeof(float)*(((FORMA_BLOCKDIM_Z-0)*(FORMA_BLOCKDIM_Y-0)*(FORMA_BLOCKDIM_X-0)));
  return SMemSize;
}

__global__ void __kernel___forma_kernel__1__(float * __restrict__ input, int L, int M, int N, float * __restrict__ __copy_arr_0__, float * __restrict__ __copy_arr_1__, float * __restrict__ __copy_arr_2__, int FORMA_BLOCKDIM_X, int FORMA_BLOCKDIM_Y, int FORMA_BLOCKDIM_Z, float * __restrict__ __var_1__){
  int __FORMA_SHARED_MEM_OFFSET__ = 0;
  float* __tilevar_0__ = (float*)(__FORMA_SHARED_MEM__+__FORMA_SHARED_MEM_OFFSET__);
  __FORMA_SHARED_MEM_OFFSET__ += sizeof(float)*(((FORMA_BLOCKDIM_Z-0)*(FORMA_BLOCKDIM_Y-0)*(FORMA_BLOCKDIM_X-0)));
  float* __tilevar_1__ = (float*)(__FORMA_SHARED_MEM__+__FORMA_SHARED_MEM_OFFSET__);
  __FORMA_SHARED_MEM_OFFSET__ += sizeof(float)*(((FORMA_BLOCKDIM_Z-0)*(FORMA_BLOCKDIM_Y-0)*(FORMA_BLOCKDIM_X-0)));
  float * __tilevar_2__ = __tilevar_0__;
  float * __tilevar_3__ = __tilevar_1__;
  float * __tilevar_4__ = __tilevar_0__;
  float * __tilevar_5__ = __tilevar_1__;
  int __iter_0__ = (int)(blockIdx.x)*((int)(FORMA_BLOCKDIM_X)+GAPX) + (int)(FORMA_BLOCKDIM_X);
  int __iter_1__ = (int)(blockIdx.y)*((int)(FORMA_BLOCKDIM_Y));
  int __iter_2__ = (int)(blockIdx.z)*((int)(FORMA_BLOCKDIM_Z));

  int __iter_3__ = FORMA_MAX(__iter_2__,0) + (int)(threadIdx.z) ; 
  for( ; __iter_3__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-1),(L-1)) ; __iter_3__+=(int)(blockDim.z) ){
    int __iter_4__ = FORMA_MAX(__iter_1__,0) + (int)(threadIdx.y) ; 
    if(__iter_4__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-1),(M-1))) {
      int __iter_5__ = FORMA_MAX(__iter_0__-2,0) + (int)(threadIdx.x) ; 
      if( __iter_5__ <= FORMA_MIN(((__iter_0__+GAPX+2)-1),(N-1)) ){
        __tilevar_2__[__iter_5__+(EXTENT-__iter_0__)+(FORMA_BLOCKDIM_X-0)*(__iter_4__+(0-__iter_1__)+(FORMA_BLOCKDIM_Y-0)*(__iter_3__+(0-__iter_2__)))] = input[__iter_5__+(N-0)*(__iter_4__+(M-0)*(__iter_3__))];
      }
    }
  }
  __syncthreads();
  int __iter_6__ = FORMA_MAX((__iter_2__+1),1) + (int)(threadIdx.z) ; 
  for( ; __iter_6__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-2),(L-2)) ; __iter_6__+=(int)(blockDim.z) ){
    int __iter_7__ = FORMA_MAX((__iter_1__+1),1) + (int)(threadIdx.y) ; 
    if(__iter_7__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-2),(M-2))) {
      int __iter_8__ = FORMA_MAX((__iter_0__-1),1) + (int)(threadIdx.x) ; 
      if( __iter_8__ <= FORMA_MIN(((__iter_0__+GAPX+1)-1),(N-2)) ){
        float __temp_3__ = (__tilevar_2__[__iter_8__+(1)+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_7__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_6__+(0-(__iter_2__+0))))]);
        float __temp_7__ = (__tilevar_2__[__iter_8__+(-1)+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_7__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_6__+(0-(__iter_2__+0))))]);
        float __temp_8__ = (0.161000f * __temp_3__ + 0.162000f * __temp_7__);
        float __temp_12__ = (__tilevar_2__[__iter_8__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_7__+(1)+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_6__+(0-(__iter_2__+0))))]);
        float __temp_13__ = (__temp_8__ + 0.163000f * __temp_12__);
        float __temp_17__ = (__tilevar_2__[__iter_8__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_7__+(-1)+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_6__+(0-(__iter_2__+0))))]);
        float __temp_18__ = (__temp_13__ + 0.164000f * __temp_17__);
        float __temp_22__ = (__tilevar_2__[__iter_8__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_7__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_6__+(1)+(0-(__iter_2__+0))))]);
        float __temp_23__ = (__temp_18__ + 0.165000f * __temp_22__);
        float __temp_27__ = (__tilevar_2__[__iter_8__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_7__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_6__+(-1)+(0-(__iter_2__+0))))]);
        float __temp_28__ = (__temp_23__ + 0.166000f * __temp_27__);
        float __temp_32__ = (__tilevar_2__[__iter_8__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_7__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*( __iter_6__+(0-(__iter_2__+0))))]);
        float __temp_33__ = (__temp_28__ - 1.670000f * __temp_32__);
        __tilevar_3__[__iter_8__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_7__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_6__+(0-(__iter_2__+0))))] = __temp_33__;
      }
    }
  }
  __syncthreads();
  int __iter_9__ = FORMA_MAX((__iter_2__+1),1) + (int)(threadIdx.z) ; 
  for( ; __iter_9__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-2),(L-2)) ; __iter_9__ += (int)(blockDim.z) ){
    int __iter_10__ = FORMA_MAX((__iter_1__+1),1) + (int)(threadIdx.y) ; 
    if(__iter_10__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-2),(M-2))) {
      int __iter_11__ = FORMA_MAX((__iter_0__-1),1) + (int)(threadIdx.x) ; 
      if( __iter_11__ <= FORMA_MIN(((__iter_0__+GAPX+1)-1),(N-2)) ){
        if (__iter_9__ < (FORMA_MAX((__iter_2__+1),1)+2) || __iter_9__ > (FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-2),(L-2))-2) || __iter_10__ < (FORMA_MAX((__iter_1__+1),1)+2) || __iter_10__ > (FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-2),(M-2))-2)) {
          __copy_arr_0__[__iter_11__+(N-0)*(__iter_10__+(M-0)*(__iter_9__))] = __tilevar_3__[__iter_11__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_10__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_9__+(0-(__iter_2__+0))))];
        }
      }
    }
  }
  __syncthreads();
  __iter_9__ = FORMA_MAX((__iter_2__+1),1) + (int)(threadIdx.z) ;
  for( ; __iter_9__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-2),(L-2)) ; __iter_9__ += (int)(blockDim.z) ){
    int __iter_10__ = FORMA_MAX((__iter_1__+1),1) + (int)(threadIdx.y) ;
    if(__iter_10__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-2),(M-2))) {
      int __iter_11__ = FORMA_MAX((__iter_0__-3),1) + (int)(threadIdx.x) ;
      if( __iter_11__ <= FORMA_MIN(((__iter_0__+GAPX+3)-1),(N-2)) ){
        if (__iter_11__ < FORMA_MAX((__iter_0__-1),1) || __iter_11__ > FORMA_MIN(((__iter_0__+GAPX+1)-1),(N-2))) {
         __tilevar_3__[__iter_11__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_10__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_9__+(0-(__iter_2__+0))))] = __copy_arr_0__[__iter_11__+(N-0)*(__iter_10__+(M-0)*(__iter_9__))];
        }
      }
    }
  }
 __syncthreads();
  int __iter_15__ = FORMA_MAX((__iter_2__+2),1) + (int)(threadIdx.z) ; 
  for( ; __iter_15__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-3),(L-2)) ; __iter_15__ += (int)(blockDim.z) ){
    int __iter_16__ = FORMA_MAX((__iter_1__+2),1) + (int)(threadIdx.y) ; 
    if(__iter_16__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-3),(M-2))) {
      int __iter_17__ = FORMA_MAX((__iter_0__-2),1) + (int)(threadIdx.x) ; 
      if( __iter_17__ <= FORMA_MIN(((__iter_0__+GAPX+2)-1),(N-2)) ){
        float __temp_50__ = (__tilevar_3__[__iter_17__+(1)+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_16__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_15__+(0-(__iter_2__+0))))]);
        float __temp_54__ = (__tilevar_3__[__iter_17__+(-1)+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_16__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_15__+(0-(__iter_2__+0))))]);
        float __temp_55__ = (0.161000f * __temp_50__ + 0.162000f * __temp_54__);
        float __temp_59__ = (__tilevar_3__[__iter_17__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_16__+(1)+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_15__+(0-(__iter_2__+0))))]);
        float __temp_60__ = (__temp_55__ + 0.163000f * __temp_59__);
        float __temp_64__ = (__tilevar_3__[__iter_17__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_16__+(-1)+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_15__+(0-(__iter_2__+0))))]);
        float __temp_65__ = (__temp_60__ + 0.164000f * __temp_64__);
        float __temp_69__ = (__tilevar_3__[__iter_17__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_16__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_15__+(1)+(0-(__iter_2__+0))))]);
        float __temp_70__ = (__temp_65__ + 0.165000f * __temp_69__);
        float __temp_74__ = (__tilevar_3__[__iter_17__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_16__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_15__+(-1)+(0-(__iter_2__+0))))]);
        float __temp_75__ = (__temp_70__ + 0.166000f * __temp_74__);
        float __temp_79__ = (__tilevar_3__[__iter_17__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_16__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_15__+(0-(__iter_2__+0))))]);
        float __temp_80__ = (__temp_75__ - 1.670000f * __temp_79__);
        __tilevar_4__[__iter_17__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_16__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_15__+(0-(__iter_2__+0))))] = __temp_80__;
      }
    }
  }
  __syncthreads();
  int __iter_18__ = FORMA_MAX((__iter_2__+2),1) + (int)(threadIdx.z) ; 
  for( ; __iter_18__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-3),(L-2)) ; __iter_18__ += (int)(blockDim.z) ){
    int __iter_19__ = FORMA_MAX((__iter_1__+2),1) + (int)(threadIdx.y) ; 
    if(__iter_19__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-3),(M-2))) {
      int __iter_20__ = FORMA_MAX((__iter_0__-2),1) + (int)(threadIdx.x) ; 
      if( __iter_20__ <= FORMA_MIN(((__iter_0__+GAPX+2)-1),(N-2)) ){
        if (__iter_18__ < (FORMA_MAX((__iter_2__+2),1)+2) || __iter_18__ > (FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-3),(L-2))-2) || __iter_19__ < (FORMA_MAX((__iter_1__+2),1)+2) || __iter_19__ > (FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-3),(M-2))-2)) {
          __copy_arr_1__[__iter_20__+(N-0)*(__iter_19__+(M-0)*(__iter_18__))] = __tilevar_4__[__iter_20__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_19__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_18__+(0-(__iter_2__+0))))];
        }
      }
    }
  }
  __syncthreads();
  __iter_18__ = FORMA_MAX((__iter_2__+2),1) + (int)(threadIdx.z) ; 
  for( ; __iter_18__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-3),(L-2)) ; __iter_18__ += (int)(blockDim.z) ){
    int __iter_19__ = FORMA_MAX((__iter_1__+2),1) + (int)(threadIdx.y) ; 
    if(__iter_19__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-3),(M-2))) {
      int __iter_20__ = FORMA_MAX((__iter_0__-4),1) + (int)(threadIdx.x) ; 
      if( __iter_20__ <= FORMA_MIN(((__iter_0__+GAPX+4)-1),(N-2)) ){
        if (__iter_20__ < FORMA_MAX((__iter_0__-2),1) || __iter_20__ > FORMA_MIN(((__iter_0__+GAPX+2)-1),(N-2))) {
          __tilevar_4__[__iter_20__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_19__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_18__+(0-(__iter_2__+0))))] = __copy_arr_1__[__iter_20__+(N-0)*(__iter_19__+(M-0)*(__iter_18__))];
        }
      }
    }
  }
  __syncthreads();
  int __iter_24__ = FORMA_MAX((__iter_2__+3),1) + (int)(threadIdx.z) ; 
  for( ; __iter_24__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-4),(L-2)) ; __iter_24__ += (int)(blockDim.z) ){
    int __iter_25__ = FORMA_MAX((__iter_1__+3),1) + (int)(threadIdx.y) ; 
    if(__iter_25__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-4),(M-2))) {
      int __iter_26__ = FORMA_MAX((__iter_0__-3),1) + (int)(threadIdx.x) ; 
      if( __iter_26__ <= FORMA_MIN(((__iter_0__+GAPX+3)-1),(N-2)) ){
        float __temp_94__ = (__tilevar_4__[__iter_26__+(1)+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+0))))]);
        float __temp_95__ = (__tilevar_4__[__iter_26__+(-1)+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+0))))]);
        float __temp_96__ = (0.161000f * __temp_94__ + 0.162000f * __temp_95__);
        float __temp_97__ = (__tilevar_4__[__iter_26__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(1)+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+0))))]);
        float __temp_98__ = (__temp_96__ + 0.163000f * __temp_97__);
        float __temp_99__ = (__tilevar_4__[__iter_26__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(-1)+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+0))))]);
        float __temp_100__ = (__temp_98__ + 0.164000f * __temp_99__);
        float __temp_101__ = (__tilevar_4__[__iter_26__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(1)+(0-(__iter_2__+0))))]);
        float __temp_102__ = (__temp_100__ + 0.165000f * __temp_101__);
        float __temp_103__ = (__tilevar_4__[__iter_26__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(-1)+(0-(__iter_2__+0))))]);
        float __temp_104__ = (__temp_102__ + 0.166000f * __temp_103__);
        float __temp_105__ = (__tilevar_4__[__iter_26__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+0))))]);
        float __temp_106__ = (__temp_104__ - 1.670000f * __temp_105__);
        __tilevar_5__[__iter_26__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+0))))] = __temp_106__;
      }
    }
  }
  __syncthreads();
  int __iter_27__ = FORMA_MAX((__iter_2__+3),1) + (int)(threadIdx.z) ; 
  for( ; __iter_27__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-4),(L-2)) ; __iter_27__ += (int)(blockDim.z) ){
    int __iter_28__ = FORMA_MAX((__iter_1__+3),1) + (int)(threadIdx.y) ; 
    if(__iter_28__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-4),(M-2))) {
      int __iter_29__ = FORMA_MAX((__iter_0__-3),1) + (int)(threadIdx.x) ; 
      if( __iter_29__ <= FORMA_MIN(((__iter_0__+GAPX+3)-1),(N-2)) ){
        if (__iter_27__ < (FORMA_MAX((__iter_2__+3),1)+2) || __iter_27__ > (FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-4),(L-2))-2) || __iter_28__ < (FORMA_MAX((__iter_1__+3),1)+2) || __iter_28__ > (FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-4),(M-2))-2)) {
          __copy_arr_2__[__iter_29__+(N-0)*(__iter_28__+(M-0)*(__iter_27__))] = __tilevar_5__[__iter_29__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_28__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_27__+(0-(__iter_2__+0))))];
        }
      }
    }
  }
  __syncthreads();
  __iter_27__ = FORMA_MAX((__iter_2__+3),1) + (int)(threadIdx.z) ; 
  for( ; __iter_27__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-4),(L-2)) ; __iter_27__ += (int)(blockDim.z) ){
    int __iter_28__ = FORMA_MAX((__iter_1__+3),1) + (int)(threadIdx.y) ; 
    if(__iter_28__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-4),(M-2))) {
      int __iter_29__ = FORMA_MAX((__iter_0__-5),1) + (int)(threadIdx.x) ; 
      if( __iter_29__ <= FORMA_MIN(((__iter_0__+GAPX+5)-1),(N-2)) ){
        if (__iter_29__ < FORMA_MAX((__iter_0__-3),1) || __iter_29__ > FORMA_MIN(((__iter_0__+GAPX+3)-1),(N-2))) {
          __tilevar_5__[__iter_29__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_28__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_27__+(0-(__iter_2__+0))))] = __copy_arr_2__[__iter_29__+(N-0)*(__iter_28__+(M-0)*(__iter_27__))];
        }
      }
    }
  }
  __syncthreads();
  int __iter_33__ = FORMA_MAX((__iter_2__+4),1) + (int)(threadIdx.z) ; 
  for( ; __iter_33__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-5),(L-2)) ; __iter_33__ += (int)(blockDim.z) ){
    int __iter_34__ = FORMA_MAX((__iter_1__+4),1) + (int)(threadIdx.y) ; 
    if(__iter_34__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-5),(M-2))) {
      int __iter_35__ = FORMA_MAX((__iter_0__-4),1) + (int)(threadIdx.x) ; 
      if( __iter_35__ <= FORMA_MIN(((__iter_0__+GAPX+4)-1),(N-2)) ){
        float __temp_120__ = (__tilevar_5__[__iter_35__+(1)+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+0))))]);
        float __temp_121__ = (__tilevar_5__[__iter_35__+(-1)+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+0))))]);
        float __temp_122__ = (0.161000f * __temp_120__ + 0.162000f * __temp_121__);
        float __temp_123__ = (__tilevar_5__[__iter_35__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(1)+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+0))))]);
        float __temp_124__ = (__temp_122__ + 0.163000f * __temp_123__);
        float __temp_125__ = (__tilevar_5__[__iter_35__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(-1)+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+0))))]);
        float __temp_126__ = (__temp_124__ + 0.164000f * __temp_125__);
        float __temp_127__ = (__tilevar_5__[__iter_35__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(1)+(0-(__iter_2__+0))))]);
        float __temp_128__ = (__temp_126__ + 0.165000f * __temp_127__);
        float __temp_129__ = (__tilevar_5__[__iter_35__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(-1)+(0-(__iter_2__+0))))]);
        float __temp_130__ = (__temp_128__ + 0.166000f * __temp_129__);
        float __temp_131__ = (__tilevar_5__[__iter_35__+(EXTENT-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+0))))]);
        float __temp_132__ = (__temp_130__ - 1.670000f * __temp_131__);
        __var_1__[__iter_35__+(N-0)*(__iter_34__+(M-0)*(__iter_33__))] = __temp_132__;
      }
    }
  }
}

__global__ void __kernel___forma_kernel__2__(float * __restrict__ input, int L, int M, int N, int FORMA_BLOCKDIM_X, int FORMA_BLOCKDIM_Y, int FORMA_BLOCKDIM_Z, float * __restrict__ __copy_arr_0__){
  int __iter_0__ = (int)(blockIdx.x)*(int)(FORMA_BLOCKDIM_X);
  int __iter_1__ = (int)(blockIdx.y)*(int)(FORMA_BLOCKDIM_Y);
  int __iter_2__ = (int)(blockIdx.z)*(int)(FORMA_BLOCKDIM_Z);
  int __iter_12__ = FORMA_MAX(__iter_2__,1) + (int)(threadIdx.z) ; 
  for( ; __iter_12__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-1),(L-2)) ; __iter_12__ += (int)(blockDim.z) ){
    int __iter_13__ = FORMA_MAX(__iter_1__,1) + (int)(threadIdx.y) ; 
    for( ; __iter_13__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-1),(M-2)) ; __iter_13__ += (int)(blockDim.y) ){
      int __iter_14__ = FORMA_MAX(__iter_0__,1) + (int)(threadIdx.x) ; 
      if( __iter_14__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-1),(N-2)) ){
        if (__iter_12__ < FORMA_MAX((__iter_2__+1),1) || __iter_12__ > FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-2),(L-2)) || __iter_13__ < FORMA_MAX((__iter_1__+1),1) || __iter_13__ > FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-2),(M-2))) {
          float __temp_34__ = (0.161000f * input[__iter_14__+(1)+(N-0)*(__iter_13__+(M-0)*(__iter_12__))]);
          float __temp_35__ = (0.162000f * input[__iter_14__+(-1)+(N-0)*(__iter_13__+(M-0)*(__iter_12__))]);
          float __temp_36__ = (__temp_34__ + __temp_35__);
          float __temp_37__ = (0.163000f * input[__iter_14__+(N-0)*(__iter_13__+(1)+(M-0)*(__iter_12__))]);
          float __temp_38__ = (__temp_36__ + __temp_37__);
          float __temp_39__ = (0.164000f * input[__iter_14__+(N-0)*(__iter_13__+(-1)+(M-0)*(__iter_12__))]);
          float __temp_40__ = (__temp_38__ + __temp_39__);
          float __temp_41__ = (0.165000f * input[__iter_14__+(N-0)*(__iter_13__+(M-0)*(__iter_12__+(1)))]);
          float __temp_42__ = (__temp_40__ + __temp_41__);
          float __temp_43__ = (0.166000f * input[__iter_14__+(N-0)*(__iter_13__+(M-0)*(__iter_12__+(-1)))]);
          float __temp_44__ = (__temp_42__ + __temp_43__);
          float __temp_45__ = (1.670000f * input[__iter_14__+(N-0)*(__iter_13__+(M-0)*(__iter_12__))]);
          float __temp_46__ = (__temp_44__ - __temp_45__);
          __copy_arr_0__[__iter_14__+(N-0)*(__iter_13__+(M-0)*(__iter_12__))] = __temp_46__;
        }
      }
    }
  }
}

__global__ void __kernel___forma_kernel__3__(float * __restrict__ __copy_arr_0__, int L, int M, int N, int FORMA_BLOCKDIM_X, int FORMA_BLOCKDIM_Y, int FORMA_BLOCKDIM_Z, float * __restrict__ __copy_arr_1__){
  int __iter_0__ = (int)(blockIdx.x)*(int)(FORMA_BLOCKDIM_X);
  int __iter_1__ = (int)(blockIdx.y)*(int)(FORMA_BLOCKDIM_Y);
  int __iter_2__ = (int)(blockIdx.z)*(int)(FORMA_BLOCKDIM_Z);
  int __iter_21__ = FORMA_MAX(__iter_2__,1) + (int)(threadIdx.z) ; 
  for( ; __iter_21__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-1),(L-2)) ; __iter_21__ += (int)(blockDim.z) ){
    int __iter_22__ = FORMA_MAX(__iter_1__,1) + (int)(threadIdx.y) ; 
    for( ; __iter_22__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-1),(M-2)) ; __iter_22__ += (int)(blockDim.y) ){
      int __iter_23__ = FORMA_MAX(__iter_0__,1) + (int)(threadIdx.x) ; 
      if( __iter_23__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-1),(N-2)) ){
        if (__iter_21__ < FORMA_MAX((__iter_2__+2),1) || __iter_21__ > FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-3),(L-2)) || __iter_22__ < FORMA_MAX((__iter_1__+2),1) || __iter_22__ > FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-3),(M-2))) {
          float __temp_81__ = (0.161000f * __copy_arr_0__[__iter_23__+(1)+(N-0)*(__iter_22__+(M-0)*(__iter_21__))]);
          float __temp_82__ = (0.162000f * __copy_arr_0__[__iter_23__+(-1)+(N-0)*(__iter_22__+(M-0)*(__iter_21__))]);
          float __temp_83__ = (__temp_81__ + __temp_82__);
          float __temp_84__ = (0.163000f * __copy_arr_0__[__iter_23__+(N-0)*(__iter_22__+(1)+(M-0)*(__iter_21__))]);
          float __temp_85__ = (__temp_83__ + __temp_84__);
          float __temp_86__ = (0.164000f * __copy_arr_0__[__iter_23__+(N-0)*(__iter_22__+(-1)+(M-0)*(__iter_21__))]);
          float __temp_87__ = (__temp_85__ + __temp_86__);
          float __temp_88__ = (0.165000f * __copy_arr_0__[__iter_23__+(N-0)*(__iter_22__+(M-0)*(__iter_21__+(1)))]);
          float __temp_89__ = (__temp_87__ + __temp_88__);
          float __temp_90__ = (0.166000f * __copy_arr_0__[__iter_23__+(N-0)*(__iter_22__+(M-0)*(__iter_21__+(-1)))]);
          float __temp_91__ = (__temp_89__ + __temp_90__);
          float __temp_92__ = (1.670000f * __copy_arr_0__[__iter_23__+(N-0)*(__iter_22__+(M-0)*(__iter_21__))]);
          float __temp_93__ = (__temp_91__ - __temp_92__);
          __copy_arr_1__[__iter_23__+(N-0)*(__iter_22__+(M-0)*(__iter_21__))] = __temp_93__;
        }
      }
    }
  }
}

__global__ void __kernel___forma_kernel__4__(float * __restrict__ __copy_arr_1__, int L, int M, int N, int FORMA_BLOCKDIM_X, int FORMA_BLOCKDIM_Y, int FORMA_BLOCKDIM_Z, float * __restrict__ __copy_arr_2__){
  int __iter_0__ = (int)(blockIdx.x)*(int)(FORMA_BLOCKDIM_X);
  int __iter_1__ = (int)(blockIdx.y)*(int)(FORMA_BLOCKDIM_Y);
  int __iter_2__ = (int)(blockIdx.z)*(int)(FORMA_BLOCKDIM_Z);
  int __iter_30__ = FORMA_MAX(__iter_2__,1) + (int)(threadIdx.z) ; 
  for( ; __iter_30__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-1),(L-2)) ; __iter_30__ += (int)(blockDim.z) ){
    int __iter_31__ = FORMA_MAX(__iter_1__,1) + (int)(threadIdx.y) ; 
    for( ; __iter_31__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-1),(M-2)) ; __iter_31__ += (int)(blockDim.y) ){
      int __iter_32__ = FORMA_MAX(__iter_0__,1) + (int)(threadIdx.x) ; 
      if( __iter_32__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-1),(N-2)) ){
        if (__iter_30__ < FORMA_MAX((__iter_2__+3),1) || __iter_30__ > FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-4),(L-2)) || __iter_31__ < FORMA_MAX((__iter_1__+3),1) || __iter_31__ > FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-4),(M-2))) {
          float __temp_107__ = (0.161000f * __copy_arr_1__[__iter_32__+(1)+(N-0)*(__iter_31__+(M-0)*(__iter_30__))]);
          float __temp_108__ = (0.162000f * __copy_arr_1__[__iter_32__+(-1)+(N-0)*(__iter_31__+(M-0)*(__iter_30__))]);
          float __temp_109__ = (__temp_107__ + __temp_108__);
          float __temp_110__ = (0.163000f * __copy_arr_1__[__iter_32__+(N-0)*(__iter_31__+(1)+(M-0)*(__iter_30__))]);
          float __temp_111__ = (__temp_109__ + __temp_110__);
          float __temp_112__ = (0.164000f * __copy_arr_1__[__iter_32__+(N-0)*(__iter_31__+(-1)+(M-0)*(__iter_30__))]);
          float __temp_113__ = (__temp_111__ + __temp_112__);
          float __temp_114__ = (0.165000f * __copy_arr_1__[__iter_32__+(N-0)*(__iter_31__+(M-0)*(__iter_30__+(1)))]);
          float __temp_115__ = (__temp_113__ + __temp_114__);
          float __temp_116__ = (0.166000f * __copy_arr_1__[__iter_32__+(N-0)*(__iter_31__+(M-0)*(__iter_30__+(-1)))]);
          float __temp_117__ = (__temp_115__ + __temp_116__);
          float __temp_118__ = (1.670000f * __copy_arr_1__[__iter_32__+(N-0)*(__iter_31__+(M-0)*(__iter_30__))]);
          float __temp_119__ = (__temp_117__ - __temp_118__);
          __copy_arr_2__[__iter_32__+(N-0)*(__iter_31__+(M-0)*(__iter_30__))] = __temp_119__;
        }
      }
    }
  }
}

__global__ void __kernel___forma_kernel__5__(float * __restrict__ __copy_arr_2__, int L, int M, int N, int FORMA_BLOCKDIM_X, int FORMA_BLOCKDIM_Y, int FORMA_BLOCKDIM_Z, float * __restrict__ __var_1__){
  int __iter_0__ = (int)(blockIdx.x)*(int)(FORMA_BLOCKDIM_X);
  int __iter_1__ = (int)(blockIdx.y)*(int)(FORMA_BLOCKDIM_Y);
  int __iter_2__ = (int)(blockIdx.z)*(int)(FORMA_BLOCKDIM_Z);
  int __iter_36__ = FORMA_MAX(__iter_2__,1) + (int)(threadIdx.z) ; 
  for( ; __iter_36__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-1),(L-2)) ; __iter_36__ += (int)(blockDim.z) ){
    int __iter_37__ = FORMA_MAX(__iter_1__,1) + (int)(threadIdx.y) ; 
    for( ; __iter_37__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-1),(M-2)) ; __iter_37__ += (int)(blockDim.y) ){
      int __iter_38__ = FORMA_MAX(__iter_0__,1) + (int)(threadIdx.x) ; 
      if( __iter_38__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-1),(N-2)) ){
        if (__iter_36__ < FORMA_MAX((__iter_2__+4),1) || __iter_36__ > FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-5),(L-2)) || __iter_37__ < FORMA_MAX((__iter_1__+4),1) || __iter_37__ > FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-5),(M-2))) {
          float __temp_133__ = (0.161000f * __copy_arr_2__[__iter_38__+(1)+(N-0)*(__iter_37__+(M-0)*(__iter_36__))]);
          float __temp_134__ = (0.162000f * __copy_arr_2__[__iter_38__+(-1)+(N-0)*(__iter_37__+(M-0)*(__iter_36__))]);
          float __temp_135__ = (__temp_133__ + __temp_134__);
          float __temp_136__ = (0.163000f * __copy_arr_2__[__iter_38__+(N-0)*(__iter_37__+(1)+(M-0)*(__iter_36__))]);
          float __temp_137__ = (__temp_135__ + __temp_136__);
          float __temp_138__ = (0.164000f * __copy_arr_2__[__iter_38__+(N-0)*(__iter_37__+(-1)+(M-0)*(__iter_36__))]);
          float __temp_139__ = (__temp_137__ + __temp_138__);
          float __temp_140__ = (0.165000f * __copy_arr_2__[__iter_38__+(N-0)*(__iter_37__+(M-0)*(__iter_36__+(1)))]);
          float __temp_141__ = (__temp_139__ + __temp_140__);
          float __temp_142__ = (0.166000f * __copy_arr_2__[__iter_38__+(N-0)*(__iter_37__+(M-0)*(__iter_36__+(-1)))]);
          float __temp_143__ = (__temp_141__ + __temp_142__);
          float __temp_144__ = (1.670000f * __copy_arr_2__[__iter_38__+(N-0)*(__iter_37__+(M-0)*(__iter_36__))]);
          float __temp_145__ = (__temp_143__ - __temp_144__);
          __var_1__[__iter_38__+(N-0)*(__iter_37__+(M-0)*(__iter_36__))] = __temp_145__;
        }
      }
    }
  }
}

/*Device code End */
/* Host Code Begin */
extern "C" void j3d7pt(float * h_input, int L, int M, int N, float * __var_0__){

/* Host allocation Begin */
  float * input;
  cudaMalloc(&input,sizeof(float)*((L-0)*(M-0)*(N-0)));
  Check_CUDA_Error("Allocation Error!! : input\n");
  cudaPointerAttributes ptrAttrib_h_input;
  cudaMemcpyKind memcpy_kind_h_input = cudaMemcpyHostToDevice;
  if (cudaPointerGetAttributes(&ptrAttrib_h_input, h_input) == cudaSuccess)
    if (ptrAttrib_h_input.memoryType == cudaMemoryTypeDevice)
      memcpy_kind_h_input = cudaMemcpyDeviceToDevice;
  cudaGetLastError();
  if( memcpy_kind_h_input != cudaMemcpyDeviceToDevice ){
    cudaMemcpy(input,h_input,sizeof(float)*((L-0)*(M-0)*(N-0)), memcpy_kind_h_input);
  }
  float * __var_1__;
  cudaMalloc(&__var_1__,sizeof(float)*((L-0)*(M-0)*(N-0)));
  Check_CUDA_Error("Allocation Error!! : __var_1__\n");
  float * __copy_arr_0__;
  cudaMalloc(&__copy_arr_0__,sizeof(float)*((L-0)*(M-0)*(N-0)));
  Check_CUDA_Error("Allocation Error!! : __copy_arr_0__\n");
  float * __copy_arr_1__;
  cudaMalloc(&__copy_arr_1__,sizeof(float)*((L-0)*(M-0)*(N-0)));
  Check_CUDA_Error("Allocation Error!! : __copy_arr_1__\n");
  float * __copy_arr_2__;
  cudaMalloc(&__copy_arr_2__,sizeof(float)*((L-0)*(M-0)*(N-0)));
  Check_CUDA_Error("Allocation Error!! : __copy_arr_2__\n");
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
  int __size_0___kernel___forma_kernel__0__ = ((N-1) - 0 ) + 1;
  int __size_1___kernel___forma_kernel__0__ = ((M-1) - 0 ) + 1;
  int __size_2___kernel___forma_kernel__0__ = ((L-1) - 0 ) + 1;
  int __block_0___kernel___forma_kernel__0__ = 32;
  int __block_1___kernel___forma_kernel__0__ = 16;
  int __block_2___kernel___forma_kernel__0__ = 12;
  dim3 __blockConfig___kernel___forma_kernel__0__(__block_0___kernel___forma_kernel__0__,__block_1___kernel___forma_kernel__0__,__block_2___kernel___forma_kernel__0__);
  int __SMemSize___kernel___forma_kernel__0__ = 0;
  __SMemSize___kernel___forma_kernel__0__ = __blockSizeToSMemSize___kernel___forma_kernel__0__(__blockConfig___kernel___forma_kernel__0__);
  int __grid_0___kernel___forma_kernel__0__ = FORMA_CEIL(__size_0___kernel___forma_kernel__0__,__blockConfig___kernel___forma_kernel__0__.x+GAPX);
  int __grid_1___kernel___forma_kernel__0__ = FORMA_CEIL(__size_1___kernel___forma_kernel__0__,__blockConfig___kernel___forma_kernel__0__.y);
  int __grid_2___kernel___forma_kernel__0__ = FORMA_CEIL(__size_2___kernel___forma_kernel__0__,__blockConfig___kernel___forma_kernel__0__.z);
  dim3 __gridConfig___kernel___forma_kernel__0__(__grid_0___kernel___forma_kernel__0__,__grid_1___kernel___forma_kernel__0__,__grid_2___kernel___forma_kernel__0__);
  dim3 unrollConfig (32,16,1);
  __kernel___forma_kernel__0__<<<__gridConfig___kernel___forma_kernel__0__, unrollConfig, __SMemSize___kernel___forma_kernel__0__>>> (input, L, M, N, __copy_arr_0__, __copy_arr_1__, __copy_arr_2__, __blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y, __blockConfig___kernel___forma_kernel__0__.z, __var_1__);
  Check_CUDA_Error("Kernel Launch Error!! : __kernel___forma_kernel__0__\n");
  __kernel___forma_kernel__1__<<<__gridConfig___kernel___forma_kernel__0__, unrollConfig, __SMemSize___kernel___forma_kernel__0__>>> (input, L, M, N, __copy_arr_0__, __copy_arr_1__, __copy_arr_2__, __blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y, __blockConfig___kernel___forma_kernel__0__.z, __var_1__);
  Check_CUDA_Error("Kernel Launch Error!! : __kernel___forma_kernel__1__\n");

  __gridConfig___kernel___forma_kernel__0__.x = FORMA_CEIL(__size_0___kernel___forma_kernel__0__,__blockConfig___kernel___forma_kernel__0__.x); 
  dim3 __blockConfig___kernel___forma_kernel__2__(__blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y/2, 1);
  __kernel___forma_kernel__2__<<<__gridConfig___kernel___forma_kernel__0__, __blockConfig___kernel___forma_kernel__2__>>> (input, L, M, N, __blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y, __blockConfig___kernel___forma_kernel__0__.z, __copy_arr_0__);
  Check_CUDA_Error("Kernel Launch Error!! : __kernel___forma_kernel__2__\n");

  dim3 __blockConfig___kernel___forma_kernel__3__(__blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y/2, 1);
  __kernel___forma_kernel__3__<<<__gridConfig___kernel___forma_kernel__0__, __blockConfig___kernel___forma_kernel__3__>>> (__copy_arr_0__, L, M, N, __blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y, __blockConfig___kernel___forma_kernel__0__.z, __copy_arr_1__);
  Check_CUDA_Error("Kernel Launch Error!! : __kernel___forma_kernel__3__\n");

  dim3 __blockConfig___kernel___forma_kernel__4__(__blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y/2, 1);
  __kernel___forma_kernel__4__<<<__gridConfig___kernel___forma_kernel__0__, __blockConfig___kernel___forma_kernel__4__>>> (__copy_arr_1__, L, M, N, __blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y, __blockConfig___kernel___forma_kernel__0__.z, __copy_arr_2__);
  Check_CUDA_Error("Kernel Launch Error!! : __kernel___forma_kernel__4__\n");

  dim3 __blockConfig___kernel___forma_kernel__5__(__blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y/2, 1);
  __kernel___forma_kernel__5__<<<__gridConfig___kernel___forma_kernel__0__, __blockConfig___kernel___forma_kernel__5__>>> (__copy_arr_2__, L, M, N, __blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y, __blockConfig___kernel___forma_kernel__0__.z, __var_1__);
  Check_CUDA_Error("Kernel Launch Error!! : __kernel___forma_kernel__5__\n");

  cudaPointerAttributes ptrAttrib___var_0__;
  cudaMemcpyKind memcpy_kind___var_0__ = cudaMemcpyDeviceToHost;
  if (cudaPointerGetAttributes(&ptrAttrib___var_0__, __var_0__) == cudaSuccess)
    if (ptrAttrib___var_0__.memoryType == cudaMemoryTypeDevice)
      memcpy_kind___var_0__ = cudaMemcpyDeviceToDevice;
  cudaGetLastError();
  cudaMemcpy(__var_0__,__var_1__, sizeof(float)*((L-0)*(M-0)*(N-0)), memcpy_kind___var_0__);
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
  cudaFree(__copy_arr_0__);
  cudaFree(__copy_arr_1__);
  cudaFree(__copy_arr_2__);
}
/*Host Free End*/
