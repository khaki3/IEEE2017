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
#define GAPX (118) 
#define EXTENT (5)

#ifndef FORMA_MAX_BLOCKDIM_0
#define FORMA_MAX_BLOCKDIM_0 1024
#endif
#ifndef FORMA_MAX_BLOCKDIM_1
#define FORMA_MAX_BLOCKDIM_1 1024
#endif
#ifndef FORMA_MAX_BLOCKDIM_2
#define FORMA_MAX_BLOCKDIM_2 1024
#endif

void Check_CUDA_Error(const char* message);
/*Texture references */
/*Shared Memory Variable */
extern __shared__ char __FORMA_SHARED_MEM__[];
/* Device code Begin */

__global__ void __kernel___forma_kernel__0__(float * __restrict__ input, int N, int M, float * __restrict__ __copy_arr_0__, float * __restrict__ __copy_arr_1__, float * __restrict__ __copy_arr_2__, int FORMA_BLOCKDIM_X, int FORMA_BLOCKDIM_Y, float * __restrict__ __var_1__){
  int __FORMA_SHARED_MEM_OFFSET__ = 0;
  float * __tilevar_2__ = (float*)(__FORMA_SHARED_MEM__+__FORMA_SHARED_MEM_OFFSET__);
  __FORMA_SHARED_MEM_OFFSET__ += sizeof(float)*(FORMA_BLOCKDIM_X);
  float * __tilevar_3__ = (float*)(__FORMA_SHARED_MEM__+__FORMA_SHARED_MEM_OFFSET__);
  __FORMA_SHARED_MEM_OFFSET__ += sizeof(float)*(FORMA_BLOCKDIM_X);
  float * __tilevar_4__ = (float*)(__FORMA_SHARED_MEM__+__FORMA_SHARED_MEM_OFFSET__);
  __FORMA_SHARED_MEM_OFFSET__ += sizeof(float)*(FORMA_BLOCKDIM_X);
  float * __tilevar_5__ = (float*)(__FORMA_SHARED_MEM__+__FORMA_SHARED_MEM_OFFSET__);
  __FORMA_SHARED_MEM_OFFSET__ += sizeof(float)*(FORMA_BLOCKDIM_X);

  int __iter_0__ = (int)(blockIdx.x)*((int)FORMA_BLOCKDIM_X + GAPX);
  float t2=0.0f, t3=0.0f, t4=0.0f, t5=0.0f;
  float b2=0.0f, b3=0.0f, b4=0.0f, b5=0.0f;

  // Initialize the value
  int __iter_3__ = FORMA_MAX(__iter_0__,0) + (int)(threadIdx.x) ;
  if(__iter_3__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-1),(M-1))){
     __tilevar_2__[__iter_3__-__iter_0__] = input[__iter_3__+M*0];
     t2 = input[__iter_3__+M*1];
  }
  // Rest of the computation
  for (int __iter_1__ = 1; __iter_1__ <= N-1; __iter_1__++) {
    if(__iter_3__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-1),(M-1))){
      b2 = __tilevar_2__[__iter_3__-__iter_0__];
      __tilevar_2__[__iter_3__-__iter_0__] = t2; 
      t2 = input[__iter_3__+M*(__iter_1__+1)];
    }
    __syncthreads();
    if(__iter_3__ >= FORMA_MAX((__iter_0__+1),1) &  __iter_3__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-2),(M-2)) ){
        float __temp_0__ = (__tilevar_2__[__iter_3__-__iter_0__] - b2);
        float __temp_1__ = (__tilevar_2__[__iter_3__-__iter_0__] - b2);
        float __temp_2__ = (__temp_0__ * __temp_1__);
        float __temp_3__ = (0.000100f + __temp_2__);
        float __temp_4__ = (__tilevar_2__[__iter_3__-__iter_0__] - t2);
        float __temp_5__ = (__tilevar_2__[__iter_3__-__iter_0__] - t2);
        float __temp_6__ = (__temp_4__ * __temp_5__);
        float __temp_7__ = (__temp_3__ + __temp_6__);
        float __temp_8__ = (__tilevar_2__[__iter_3__-__iter_0__] - __tilevar_2__[__iter_3__+1-__iter_0__]);
        float __temp_9__ = (__tilevar_2__[__iter_3__-__iter_0__] - __tilevar_2__[__iter_3__+1-__iter_0__]);
        float __temp_10__ = (__temp_8__ * __temp_9__);
        float __temp_11__ = (__temp_7__ + __temp_10__);
        float __temp_12__ = (__tilevar_2__[__iter_3__-__iter_0__] - __tilevar_2__[__iter_3__-1-__iter_0__]);
        float __temp_13__ = (__tilevar_2__[__iter_3__-__iter_0__] - __tilevar_2__[__iter_3__-1-__iter_0__]);
        float __temp_14__ = (__temp_12__ * __temp_13__);
        float __temp_15__ = (__temp_11__ + __temp_14__);
        float __temp_16__ = sqrt(__temp_15__);
        float __temp_17__ = (1.000000f / __temp_16__);
        float __temp_18__ = (__tilevar_2__[__iter_3__-__iter_0__] + __temp_17__);
        b3 = __tilevar_3__[__iter_3__-__iter_0__];
        __tilevar_3__[__iter_3__-__iter_0__] = t3;
        t3 = __temp_18__;
    }
    if(__iter_3__ >= FORMA_MAX((__iter_0__+1),1) & __iter_3__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-2),(M-2)) ){
      if (__iter_3__ < (FORMA_MAX((__iter_0__+1),1)+2) | __iter_3__ > (FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-2),(M-2))-2)) {
        __copy_arr_0__[__iter_1__+(M)*(__iter_3__)] = t3;
      }
    }
    __syncthreads();
    if(__iter_3__ >= FORMA_MAX((__iter_0__+2),1) & __iter_3__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-3),(M-2)) ){
        float __temp_0__ = (__tilevar_3__[__iter_3__-__iter_0__] - b3);
        float __temp_1__ = (__tilevar_3__[__iter_3__-__iter_0__] - b3);
        float __temp_2__ = (__temp_0__ * __temp_1__);
        float __temp_3__ = (0.000100f + __temp_2__);
        float __temp_4__ = (__tilevar_3__[__iter_3__-__iter_0__] - t3);
        float __temp_5__ = (__tilevar_3__[__iter_3__-__iter_0__] - t3);
        float __temp_6__ = (__temp_4__ * __temp_5__);
        float __temp_7__ = (__temp_3__ + __temp_6__);
        float __temp_8__ = (__tilevar_3__[__iter_3__-__iter_0__] - __tilevar_3__[__iter_3__+1-__iter_0__]);
        float __temp_9__ = (__tilevar_3__[__iter_3__-__iter_0__] - __tilevar_3__[__iter_3__+1-__iter_0__]);
        float __temp_10__ = (__temp_8__ * __temp_9__);
        float __temp_11__ = (__temp_7__ + __temp_10__);
        float __temp_12__ = (__tilevar_3__[__iter_3__-__iter_0__] - __tilevar_3__[__iter_3__-1-__iter_0__]);
        float __temp_13__ = (__tilevar_3__[__iter_3__-__iter_0__] - __tilevar_3__[__iter_3__-1-__iter_0__]);
        float __temp_14__ = (__temp_12__ * __temp_13__);
        float __temp_15__ = (__temp_11__ + __temp_14__);
        float __temp_16__ = sqrt(__temp_15__);
        float __temp_17__ = (1.000000f / __temp_16__);
        float __temp_18__ = (__tilevar_3__[__iter_3__-__iter_0__] + __temp_17__);
        b4 = __tilevar_4__[__iter_3__-__iter_0__];
        __tilevar_4__[__iter_3__-__iter_0__] = t4;
        t4 = __temp_18__;
    }
    if(__iter_3__ >= FORMA_MAX((__iter_0__+2),1) &  __iter_3__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-3),(M-2)) ){
      if (__iter_3__ < (FORMA_MAX((__iter_0__+2),1)+2) | __iter_3__ > (FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-3),(M-2))-2)) {
        __copy_arr_1__[__iter_1__+(M)*(__iter_3__)] = t4;
      }
    }
    __syncthreads();
    if(__iter_3__ >= FORMA_MAX((__iter_0__+3),1) & __iter_3__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-4),(M-2)) ){
        float __temp_0__ = (__tilevar_4__[__iter_3__-__iter_0__] - b4);
        float __temp_1__ = (__tilevar_4__[__iter_3__-__iter_0__] - b4);
        float __temp_2__ = (__temp_0__ * __temp_1__);
        float __temp_3__ = (0.000100f + __temp_2__);
        float __temp_4__ = (__tilevar_4__[__iter_3__-__iter_0__] - t4);
        float __temp_5__ = (__tilevar_4__[__iter_3__-__iter_0__] - t4);
        float __temp_6__ = (__temp_4__ * __temp_5__);
        float __temp_7__ = (__temp_3__ + __temp_6__);
        float __temp_8__ = (__tilevar_4__[__iter_3__-__iter_0__] - __tilevar_4__[__iter_3__+1-__iter_0__]);
        float __temp_9__ = (__tilevar_4__[__iter_3__-__iter_0__] - __tilevar_4__[__iter_3__+1-__iter_0__]);
        float __temp_10__ = (__temp_8__ * __temp_9__);
        float __temp_11__ = (__temp_7__ + __temp_10__);
        float __temp_12__ = (__tilevar_4__[__iter_3__-__iter_0__] - __tilevar_4__[__iter_3__-1-__iter_0__]);
        float __temp_13__ = (__tilevar_4__[__iter_3__-__iter_0__] - __tilevar_4__[__iter_3__-1-__iter_0__]);
        float __temp_14__ = (__temp_12__ * __temp_13__);
        float __temp_15__ = (__temp_11__ + __temp_14__);
        float __temp_16__ = sqrt(__temp_15__);
        float __temp_17__ = (1.000000f / __temp_16__);
        float __temp_18__ = (__tilevar_4__[__iter_3__-__iter_0__] + __temp_17__);
        b5 = __tilevar_5__[__iter_3__-__iter_0__];
        __tilevar_5__[__iter_3__-__iter_0__] = t5;
        t5 = __temp_18__;
    }
    if(__iter_3__ >= FORMA_MAX((__iter_0__+3),1) & __iter_3__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-4),(M-2)) ){
      if (__iter_3__ < (FORMA_MAX((__iter_0__+3),1)+2) | __iter_3__ > (FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-4),(M-2))-2)) {
        __copy_arr_2__[__iter_1__+(M)*(__iter_3__)] = t5;
      }
    }
    __syncthreads();
    if(__iter_3__ >= FORMA_MAX((__iter_0__+4),1) & __iter_3__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-5),(M-2)) ){
        float __temp_0__ = (__tilevar_5__[__iter_3__-__iter_0__] - b5);
        float __temp_1__ = (__tilevar_5__[__iter_3__-__iter_0__] - b5);
        float __temp_2__ = (__temp_0__ * __temp_1__);
        float __temp_3__ = (0.000100f + __temp_2__);
        float __temp_4__ = (__tilevar_5__[__iter_3__-__iter_0__] - t5);
        float __temp_5__ = (__tilevar_5__[__iter_3__-__iter_0__] - t5);
        float __temp_6__ = (__temp_4__ * __temp_5__);
        float __temp_7__ = (__temp_3__ + __temp_6__);
        float __temp_8__ = (__tilevar_5__[__iter_3__-__iter_0__] - __tilevar_5__[__iter_3__+1-__iter_0__]);
        float __temp_9__ = (__tilevar_5__[__iter_3__-__iter_0__] - __tilevar_5__[__iter_3__+1-__iter_0__]);
        float __temp_10__ = (__temp_8__ * __temp_9__);
        float __temp_11__ = (__temp_7__ + __temp_10__);
        float __temp_12__ = (__tilevar_5__[__iter_3__-__iter_0__] - __tilevar_5__[__iter_3__-1-__iter_0__]);
        float __temp_13__ = (__tilevar_5__[__iter_3__-__iter_0__] - __tilevar_5__[__iter_3__-1-__iter_0__]);
        float __temp_14__ = (__temp_12__ * __temp_13__);
        float __temp_15__ = (__temp_11__ + __temp_14__);
        float __temp_16__ = sqrt(__temp_15__);
        float __temp_17__ = (1.000000f / __temp_16__);
        float __temp_18__ = (__tilevar_5__[__iter_3__-__iter_0__] + __temp_17__);
        __var_1__[__iter_3__+(M)*FORMA_MAX(__iter_1__-3,0)] = __temp_18__;
    }
  }
}

int __blockSizeToSMemSize___kernel___forma_kernel__0__(dim3 blockDim){
  int FORMA_BLOCKDIM_X = (int)(blockDim.x);
  int SMemSize = 0;
  SMemSize += sizeof(float)*(4*FORMA_BLOCKDIM_X);
  return SMemSize;
}

__global__ void __kernel___forma_kernel__1__(float * __restrict__ input, int N, int M, float * __restrict__ __copy_arr_0__, float * __restrict__ __copy_arr_1__, float * __restrict__ __copy_arr_2__, int FORMA_BLOCKDIM_X, int FORMA_BLOCKDIM_Y, float * __restrict__ __var_1__){
  int __FORMA_SHARED_MEM_OFFSET__ = 0;
  float * __tilevar_2__ = (float*)(__FORMA_SHARED_MEM__+__FORMA_SHARED_MEM_OFFSET__);
  __FORMA_SHARED_MEM_OFFSET__ += sizeof(float)*(FORMA_BLOCKDIM_X);
  float * __tilevar_3__ = (float*)(__FORMA_SHARED_MEM__+__FORMA_SHARED_MEM_OFFSET__);
  __FORMA_SHARED_MEM_OFFSET__ += sizeof(float)*(FORMA_BLOCKDIM_X);
  float * __tilevar_4__ = (float*)(__FORMA_SHARED_MEM__+__FORMA_SHARED_MEM_OFFSET__);
  __FORMA_SHARED_MEM_OFFSET__ += sizeof(float)*(FORMA_BLOCKDIM_X);
  float * __tilevar_5__ = (float*)(__FORMA_SHARED_MEM__+__FORMA_SHARED_MEM_OFFSET__);
  __FORMA_SHARED_MEM_OFFSET__ += sizeof(float)*(FORMA_BLOCKDIM_X);
 
  int __iter_0__ = (int)(blockIdx.x)*((int)FORMA_BLOCKDIM_X + GAPX) + (int)FORMA_BLOCKDIM_X;
  float t2=0.0f, t3=0.0f, t4=0.0f, t5=0.0f;
  float b2=0.0f, b3=0.0f, b4=0.0f, b5=0.0f;

  // Initialize the values
  int __iter_3__ = FORMA_MAX(__iter_0__-EXTENT,0) + (int)(threadIdx.x) ;
  if (__iter_3__ >= FORMA_MAX(__iter_0__-2,0) & __iter_3__ <= FORMA_MIN(((__iter_0__+GAPX+2)-1),(M-1))){
    __tilevar_2__[__iter_3__+(EXTENT-__iter_0__)] = input[__iter_3__+(M)*(0)];
    t2 = input[__iter_3__+(M)*(1)];
  }
  // Rest of the computation
  for (int __iter_1__ = 1; __iter_1__ <= N-1; __iter_1__++) {
    if(__iter_3__ >= FORMA_MAX(__iter_0__-2,0) & __iter_3__ <= FORMA_MIN(((__iter_0__+GAPX+2)-1),(M-1))){
      b2 = __tilevar_2__[__iter_3__+(EXTENT-__iter_0__)];
      __tilevar_2__[__iter_3__+(EXTENT-__iter_0__)] = t2;
      t2 = input[__iter_3__+(M)*(__iter_1__+1)];
    }
    __syncthreads();
    if(__iter_3__ >= FORMA_MAX((__iter_0__-1),1) & __iter_3__ <= FORMA_MIN(((__iter_0__+GAPX+1)-1),(M-2)) ){
	float __temp_0__ = (__tilevar_2__[__iter_3__+EXTENT-__iter_0__] - b2);
        float __temp_1__ = (__tilevar_2__[__iter_3__+EXTENT-__iter_0__] - b2);
        float __temp_2__ = (__temp_0__ * __temp_1__);
        float __temp_3__ = (0.000100f + __temp_2__);
        float __temp_4__ = (__tilevar_2__[__iter_3__+EXTENT-__iter_0__] - t2);
        float __temp_5__ = (__tilevar_2__[__iter_3__+EXTENT-__iter_0__] - t2);
        float __temp_6__ = (__temp_4__ * __temp_5__);
        float __temp_7__ = (__temp_3__ + __temp_6__);
        float __temp_8__ = (__tilevar_2__[__iter_3__+EXTENT-__iter_0__] - __tilevar_2__[__iter_3__+1+EXTENT-__iter_0__]);
        float __temp_9__ = (__tilevar_2__[__iter_3__+EXTENT-__iter_0__] - __tilevar_2__[__iter_3__+1+EXTENT-__iter_0__]);
        float __temp_10__ = (__temp_8__ * __temp_9__);
        float __temp_11__ = (__temp_7__ + __temp_10__);
        float __temp_12__ = (__tilevar_2__[__iter_3__+EXTENT-__iter_0__] - __tilevar_2__[__iter_3__-1+EXTENT-__iter_0__]);
        float __temp_13__ = (__tilevar_2__[__iter_3__+EXTENT-__iter_0__] - __tilevar_2__[__iter_3__-1+EXTENT-__iter_0__]);
        float __temp_14__ = (__temp_12__ * __temp_13__);
        float __temp_15__ = (__temp_11__ + __temp_14__);
        float __temp_16__ = sqrt(__temp_15__);
        float __temp_17__ = (1.000000f / __temp_16__);
        float __temp_18__ = (__tilevar_2__[__iter_3__+EXTENT-__iter_0__] + __temp_17__);
        b3 = __tilevar_3__[__iter_3__+EXTENT-__iter_0__];
        __tilevar_3__[__iter_3__+EXTENT-__iter_0__] = t3;
        t3 = __temp_18__;
    }
    if (__iter_3__ >= FORMA_MAX((__iter_0__-3),1) & __iter_3__ <= FORMA_MIN(((__iter_0__+GAPX+3)-1),(M-2)) & (__iter_3__ < FORMA_MAX((__iter_0__-1),1) | __iter_3__ > FORMA_MIN(((__iter_0__+GAPX+1)-1),(M-2)))) {
      b3 = __copy_arr_0__[__iter_1__-2+(M)*(__iter_3__)];
      __tilevar_3__[__iter_3__+(EXTENT-__iter_0__)] = __copy_arr_0__[__iter_1__-1+(M)*(__iter_3__)];
      t3 = __copy_arr_0__[__iter_1__+(M)*(__iter_3__)]; 
    }
    __syncthreads();
    if(__iter_3__ >= FORMA_MAX((__iter_0__-2),1) & __iter_3__ <= FORMA_MIN(((__iter_0__+GAPX+2)-1),(M-2)) ){
        float __temp_0__ = (__tilevar_3__[__iter_3__+EXTENT-__iter_0__] - b3);
        float __temp_1__ = (__tilevar_3__[__iter_3__+EXTENT-__iter_0__] - b3);
        float __temp_2__ = (__temp_0__ * __temp_1__);
        float __temp_3__ = (0.000100f + __temp_2__);
        float __temp_4__ = (__tilevar_3__[__iter_3__+EXTENT-__iter_0__] - t3);
        float __temp_5__ = (__tilevar_3__[__iter_3__+EXTENT-__iter_0__] - t3);
        float __temp_6__ = (__temp_4__ * __temp_5__);
        float __temp_7__ = (__temp_3__ + __temp_6__);
        float __temp_8__ = (__tilevar_3__[__iter_3__+EXTENT-__iter_0__] - __tilevar_3__[__iter_3__+1+EXTENT-__iter_0__]);
        float __temp_9__ = (__tilevar_3__[__iter_3__+EXTENT-__iter_0__] - __tilevar_3__[__iter_3__+1+EXTENT-__iter_0__]);
        float __temp_10__ = (__temp_8__ * __temp_9__);
        float __temp_11__ = (__temp_7__ + __temp_10__);
        float __temp_12__ = (__tilevar_3__[__iter_3__+EXTENT-__iter_0__] - __tilevar_3__[__iter_3__-1+EXTENT-__iter_0__]);
        float __temp_13__ = (__tilevar_3__[__iter_3__+EXTENT-__iter_0__] - __tilevar_3__[__iter_3__-1+EXTENT-__iter_0__]);
        float __temp_14__ = (__temp_12__ * __temp_13__);
        float __temp_15__ = (__temp_11__ + __temp_14__);
        float __temp_16__ = sqrt(__temp_15__);
        float __temp_17__ = (1.000000f / __temp_16__);
        float __temp_18__ = (__tilevar_3__[__iter_3__+EXTENT-__iter_0__] + __temp_17__);
        b4 = __tilevar_4__[__iter_3__+EXTENT-__iter_0__];
        __tilevar_4__[__iter_3__+EXTENT-__iter_0__] = t4;
        t4 = __temp_18__;
    }
    if (__iter_3__ >= FORMA_MAX((__iter_0__-4),1) & __iter_3__ <= FORMA_MIN(((__iter_0__+GAPX+4)-1),(M-2)) & (__iter_3__ < FORMA_MAX((__iter_0__-2),1) | __iter_3__ > FORMA_MIN(((__iter_0__+GAPX+2)-1),(M-2)))) {
      b4 = __copy_arr_1__[__iter_1__-2+(M)*(__iter_3__)];
      __tilevar_4__[__iter_3__+(EXTENT-__iter_0__)] = __copy_arr_1__[__iter_1__-1+(M)*(__iter_3__)];
      t4 = __copy_arr_1__[__iter_1__+(M)*(__iter_3__)];
    }
    __syncthreads();
    if(__iter_3__ >= FORMA_MAX((__iter_0__-3),1) & __iter_3__ <= FORMA_MIN(((__iter_0__+GAPX+3)-1),(M-2)) ){
	float __temp_0__ = (__tilevar_4__[__iter_3__+EXTENT-__iter_0__] - b4);
        float __temp_1__ = (__tilevar_4__[__iter_3__+EXTENT-__iter_0__] - b4);
        float __temp_2__ = (__temp_0__ * __temp_1__);
        float __temp_3__ = (0.000100f + __temp_2__);
        float __temp_4__ = (__tilevar_4__[__iter_3__+EXTENT-__iter_0__] - t4);
        float __temp_5__ = (__tilevar_4__[__iter_3__+EXTENT-__iter_0__] - t4);
        float __temp_6__ = (__temp_4__ * __temp_5__);
        float __temp_7__ = (__temp_3__ + __temp_6__);
        float __temp_8__ = (__tilevar_4__[__iter_3__+EXTENT-__iter_0__] - __tilevar_4__[__iter_3__+1+EXTENT-__iter_0__]);
        float __temp_9__ = (__tilevar_4__[__iter_3__+EXTENT-__iter_0__] - __tilevar_4__[__iter_3__+1+EXTENT-__iter_0__]);
        float __temp_10__ = (__temp_8__ * __temp_9__);
        float __temp_11__ = (__temp_7__ + __temp_10__);
        float __temp_12__ = (__tilevar_4__[__iter_3__+EXTENT-__iter_0__] - __tilevar_4__[__iter_3__-1+EXTENT-__iter_0__]);
        float __temp_13__ = (__tilevar_4__[__iter_3__+EXTENT-__iter_0__] - __tilevar_4__[__iter_3__-1+EXTENT-__iter_0__]);
        float __temp_14__ = (__temp_12__ * __temp_13__);
        float __temp_15__ = (__temp_11__ + __temp_14__);
        float __temp_16__ = sqrt(__temp_15__);
        float __temp_17__ = (1.000000f / __temp_16__);
        float __temp_18__ = (__tilevar_4__[__iter_3__+EXTENT-__iter_0__] + __temp_17__);
        b5 = __tilevar_5__[__iter_3__+EXTENT-__iter_0__];
        __tilevar_5__[__iter_3__+EXTENT-__iter_0__] = t5;
        t5 = __temp_18__;
    }
    if (__iter_3__ >= FORMA_MAX((__iter_0__-5),1) & __iter_3__ <= FORMA_MIN(((__iter_0__+GAPX+5)-1),(M-2)) & (__iter_3__ < FORMA_MAX((__iter_0__-3),1) | __iter_3__ > FORMA_MIN(((__iter_0__+GAPX+3)-1),(M-2)))) {
      b5 = __copy_arr_2__[__iter_1__-2+(M)*(__iter_3__)];
      __tilevar_5__[__iter_3__+(EXTENT-__iter_0__)] = __copy_arr_2__[__iter_1__-1+(M)*(__iter_3__)];
      t5 = __copy_arr_2__[__iter_1__+(M)*(__iter_3__)];
    }
    __syncthreads();
    if( __iter_3__ >= FORMA_MAX((__iter_0__-4),1) & __iter_3__ <= FORMA_MIN(((__iter_0__+GAPX+4)-1),(M-2)) ){
        float __temp_0__ = (__tilevar_5__[__iter_3__+EXTENT-__iter_0__] - b5);
        float __temp_1__ = (__tilevar_5__[__iter_3__+EXTENT-__iter_0__] - b5);
        float __temp_2__ = (__temp_0__ * __temp_1__);
        float __temp_3__ = (0.000100f + __temp_2__);
        float __temp_4__ = (__tilevar_5__[__iter_3__+EXTENT-__iter_0__] - t5);
        float __temp_5__ = (__tilevar_5__[__iter_3__+EXTENT-__iter_0__] - t5);
        float __temp_6__ = (__temp_4__ * __temp_5__);
        float __temp_7__ = (__temp_3__ + __temp_6__);
        float __temp_8__ = (__tilevar_5__[__iter_3__+EXTENT-__iter_0__] - __tilevar_5__[__iter_3__+1+EXTENT-__iter_0__]);
        float __temp_9__ = (__tilevar_5__[__iter_3__+EXTENT-__iter_0__] - __tilevar_5__[__iter_3__+1+EXTENT-__iter_0__]);
        float __temp_10__ = (__temp_8__ * __temp_9__);
        float __temp_11__ = (__temp_7__ + __temp_10__);
        float __temp_12__ = (__tilevar_5__[__iter_3__+EXTENT-__iter_0__] - __tilevar_5__[__iter_3__-1+EXTENT-__iter_0__]);
        float __temp_13__ = (__tilevar_5__[__iter_3__+EXTENT-__iter_0__] - __tilevar_5__[__iter_3__-1+EXTENT-__iter_0__]);
        float __temp_14__ = (__temp_12__ * __temp_13__);
        float __temp_15__ = (__temp_11__ + __temp_14__);
        float __temp_16__ = sqrt(__temp_15__);
        float __temp_17__ = (1.000000f / __temp_16__);
        float __temp_18__ = (__tilevar_5__[__iter_3__+EXTENT-__iter_0__] + __temp_17__);
        __var_1__[__iter_3__+(M)*FORMA_MAX(__iter_1__-3,0)] = __temp_18__;
    }
  }
}

/*Device code End */
/* Host Code Begin */
extern "C" void gradient (float * h_input, int N, int M, float * __var_0__){

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
  float * __copy_arr_0__;
  cudaMalloc(&__copy_arr_0__,sizeof(float)*((N)*(M)));
  Check_CUDA_Error("Allocation Error!! : __copy_arr_0__\n");
  float * __copy_arr_1__;
  cudaMalloc(&__copy_arr_1__,sizeof(float)*((N)*(M)));
  Check_CUDA_Error("Allocation Error!! : __copy_arr_1__\n");
  float * __copy_arr_2__;
  cudaMalloc(&__copy_arr_2__,sizeof(float)*((N)*(M)));
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
  int __size_0___kernel___forma_kernel__0__ = ((M-1) - 0 ) + 1;
  int __block_0___kernel___forma_kernel__0__ = 128;
  int __block_1___kernel___forma_kernel__0__ = 1;
  dim3 __blockConfig___kernel___forma_kernel__0__(__block_0___kernel___forma_kernel__0__,__block_1___kernel___forma_kernel__0__);
  int __SMemSize___kernel___forma_kernel__0__ = 0;
  __SMemSize___kernel___forma_kernel__0__ = __blockSizeToSMemSize___kernel___forma_kernel__0__(__blockConfig___kernel___forma_kernel__0__);
  int __grid_0___kernel___forma_kernel__0__ = FORMA_CEIL(__size_0___kernel___forma_kernel__0__,__blockConfig___kernel___forma_kernel__0__.x+GAPX);
  int __grid_1___kernel___forma_kernel__0__ = 1;
  dim3 __gridConfig___kernel___forma_kernel__0__(__grid_0___kernel___forma_kernel__0__,__grid_1___kernel___forma_kernel__0__);
  __kernel___forma_kernel__0__<<<__gridConfig___kernel___forma_kernel__0__, __blockConfig___kernel___forma_kernel__0__, __SMemSize___kernel___forma_kernel__0__>>> (input, N, M, __copy_arr_0__, __copy_arr_1__, __copy_arr_2__, __blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y, __var_1__);
  Check_CUDA_Error("Kernel Launch Error!! : __kernel___forma_kernel__0__\n");
  __kernel___forma_kernel__1__<<<__gridConfig___kernel___forma_kernel__0__, __blockConfig___kernel___forma_kernel__0__, __SMemSize___kernel___forma_kernel__0__>>> (input, N, M, __copy_arr_0__, __copy_arr_1__, __copy_arr_2__, __blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y, __var_1__);
  Check_CUDA_Error("Kernel Launch Error!! : __kernel___forma_kernel__1__\n");

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
  cudaFree(__copy_arr_0__);
  cudaFree(__copy_arr_1__);
  cudaFree(__copy_arr_2__);
}
/*Host Free End*/
