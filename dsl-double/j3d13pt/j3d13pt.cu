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
  int __iter_0__;
  __iter_0__ = (int)(blockIdx.x)*(int)(FORMA_BLOCKDIM_X);
  int __iter_1__;
  __iter_1__ = (int)(blockIdx.y)*(int)(FORMA_BLOCKDIM_Y);
  int __iter_2__;
  __iter_2__ = (int)(blockIdx.z)*(int)(FORMA_BLOCKDIM_Z);
  int __iter_3__;
  __iter_3__ = FORMA_MAX(__iter_2__,0) + (int)(threadIdx.z) ; 
  for( ; __iter_3__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-1),(L-1)) ; __iter_3__ += (int)(blockDim.z) ){
    int __iter_4__;
    __iter_4__ = FORMA_MAX(__iter_1__,0) + (int)(threadIdx.y) ; 
    if( __iter_4__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-1),(M-1)) ){
      int __iter_5__;
      __iter_5__ = FORMA_MAX(__iter_0__,0) + (int)(threadIdx.x) ; 
      if( __iter_5__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-1),(N-1)) ){
        __tilevar_2__[__iter_5__+(0-__iter_0__)+(FORMA_BLOCKDIM_X-0)*(__iter_4__+(0-__iter_1__)+(FORMA_BLOCKDIM_Y-0)*(__iter_3__+(0-__iter_2__)))] = input[__iter_5__+(N-0)*(__iter_4__+(M-0)*(__iter_3__))];
      }
    }
  }
  __syncthreads();
  int __iter_6__;
  __iter_6__ = FORMA_MAX((__iter_2__+2),2) + (int)(threadIdx.z) ; 
  for( ; __iter_6__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-3),(L-3)) ; __iter_6__ += (int)(blockDim.z) ){
    int __iter_7__;
    __iter_7__ = FORMA_MAX((__iter_1__+2),2) + (int)(threadIdx.y) ; 
    if( __iter_7__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-3),(M-3)) ){
      int __iter_8__;
      __iter_8__ = FORMA_MAX((__iter_0__+2),2) + (int)(threadIdx.x) ; 
      if( __iter_8__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-3),(N-3)) ){
        float __temp_3__ = (__tilevar_2__[__iter_8__+2-__iter_0__+FORMA_BLOCKDIM_X*(__iter_7__-__iter_1__+FORMA_BLOCKDIM_Y*(__iter_6__-__iter_2__))]);
        float __temp_7__ = (__tilevar_2__[__iter_8__+1-__iter_0__+FORMA_BLOCKDIM_X*(__iter_7__-__iter_1__+FORMA_BLOCKDIM_Y*(__iter_6__-__iter_2__))]);
        float __temp_8__ = (0.083000f * __temp_3__ + 0.083000f * __temp_7__);
        float __temp_12__ = (__tilevar_2__[__iter_8__-1-__iter_0__+FORMA_BLOCKDIM_X*(__iter_7__-__iter_1__+FORMA_BLOCKDIM_Y*(__iter_6__-__iter_2__))]);
        float __temp_13__ = (__temp_8__ + 0.083000f * __temp_12__);
        float __temp_17__ = (__tilevar_2__[__iter_8__-2-__iter_0__+FORMA_BLOCKDIM_X*(__iter_7__-__iter_1__+FORMA_BLOCKDIM_Y*(__iter_6__-__iter_2__))]);
        float __temp_18__ = (__temp_13__ + 0.083000f * __temp_17__);
        float __temp_22__ = (__tilevar_2__[__iter_8__-__iter_0__+FORMA_BLOCKDIM_X*(__iter_7__+2-__iter_1__+FORMA_BLOCKDIM_Y*(__iter_6__-__iter_2__))]);
        float __temp_23__ = (__temp_18__ + 0.083000f * __temp_22__);
        float __temp_27__ = (__tilevar_2__[__iter_8__-__iter_0__+FORMA_BLOCKDIM_X*(__iter_7__+1-__iter_1__+FORMA_BLOCKDIM_Y*(__iter_6__-__iter_2__))]);
        float __temp_28__ = (__temp_23__ + 0.083000f * __temp_27__);
        float __temp_32__ = (__tilevar_2__[__iter_8__-__iter_0__+FORMA_BLOCKDIM_X*(__iter_7__-1-__iter_1__+FORMA_BLOCKDIM_Y*(__iter_6__-__iter_2__))]);
        float __temp_33__ = (__temp_28__ + 0.083000f * __temp_32__);
        float __temp_37__ = (__tilevar_2__[__iter_8__-__iter_0__+FORMA_BLOCKDIM_X*(__iter_7__-2-__iter_1__+FORMA_BLOCKDIM_Y*(__iter_6__-__iter_2__))]);
        float __temp_38__ = (__temp_33__ + 0.083000f * __temp_37__);
        float __temp_42__ = (__tilevar_2__[__iter_8__-__iter_0__+FORMA_BLOCKDIM_X*(__iter_7__-__iter_1__+FORMA_BLOCKDIM_Y*(__iter_6__+2-__iter_2__))]);
        float __temp_43__ = (__temp_38__ + 0.083000f * __temp_42__);
        float __temp_47__ = (__tilevar_2__[__iter_8__-__iter_0__+FORMA_BLOCKDIM_X*(__iter_7__-__iter_1__+FORMA_BLOCKDIM_Y*(__iter_6__+1-__iter_2__))]);
        float __temp_48__ = (__temp_43__ + 0.083000f * __temp_47__);
        float __temp_52__ = (__tilevar_2__[__iter_8__-__iter_0__+FORMA_BLOCKDIM_X*(__iter_7__-__iter_1__+FORMA_BLOCKDIM_Y*(__iter_6__-1-__iter_2__))]);
        float __temp_53__ = (__temp_48__ + 0.083000f * __temp_52__);
        float __temp_57__ = (__tilevar_2__[__iter_8__-__iter_0__+FORMA_BLOCKDIM_X*(__iter_7__-__iter_1__+FORMA_BLOCKDIM_Y*(__iter_6__-2-__iter_2__))]);
        float __temp_58__ = (__temp_53__ + 0.083000f * __temp_57__);
        float __temp_62__ = (__tilevar_2__[__iter_8__-__iter_0__+FORMA_BLOCKDIM_X*(__iter_7__-__iter_1__+FORMA_BLOCKDIM_Y*(__iter_6__-__iter_2__))]);
        float __temp_63__ = (__temp_58__ - 0.996000f * __temp_62__);
        __tilevar_3__[__iter_8__+(0-(__iter_0__+2))+FORMA_BLOCKDIM_X*(__iter_7__+(0-(__iter_1__+2))+FORMA_BLOCKDIM_Y*(__iter_6__+(0-(__iter_2__+2))))] = __temp_63__;

        //int __temp_0__;
        //__temp_0__ = __iter_6__;
        //int __temp_1__;
        //__temp_1__ = __iter_7__;
        //int __temp_2__;
        //__temp_2__ = __iter_8__+(2);
        //float __temp_3__;
        //__temp_3__ = (0.083000f * __tilevar_2__[__temp_2__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__temp_1__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__temp_0__+(0-(__iter_2__+0))))]);
        //int __temp_4__;
        //__temp_4__ = __iter_6__;
        //int __temp_5__;
        //__temp_5__ = __iter_7__;
        //int __temp_6__;
        //__temp_6__ = __iter_8__+(1);
        //float __temp_7__;
        //__temp_7__ = (0.083000f * __tilevar_2__[__temp_6__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__temp_5__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__temp_4__+(0-(__iter_2__+0))))]);
        //float __temp_8__;
        //__temp_8__ = (__temp_3__ + __temp_7__);
        //int __temp_9__;
        //__temp_9__ = __iter_6__;
        //int __temp_10__;
        //__temp_10__ = __iter_7__;
        //int __temp_11__;
        //__temp_11__ = __iter_8__+(-1);
        //float __temp_12__;
        //__temp_12__ = (0.083000f * __tilevar_2__[__temp_11__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__temp_10__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__temp_9__+(0-(__iter_2__+0))))]);
        //float __temp_13__;
        //__temp_13__ = (__temp_8__ + __temp_12__);
        //int __temp_14__;
        //__temp_14__ = __iter_6__;
        //int __temp_15__;
        //__temp_15__ = __iter_7__;
        //int __temp_16__;
        //__temp_16__ = __iter_8__+(-2);
        //float __temp_17__;
        //__temp_17__ = (0.083000f * __tilevar_2__[__temp_16__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__temp_15__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__temp_14__+(0-(__iter_2__+0))))]);
        //float __temp_18__;
        //__temp_18__ = (__temp_13__ + __temp_17__);
        //int __temp_19__;
        //__temp_19__ = __iter_6__;
        //int __temp_20__;
        //__temp_20__ = __iter_7__+(2);
        //int __temp_21__;
        //__temp_21__ = __iter_8__;
        //float __temp_22__;
        //__temp_22__ = (0.083000f * __tilevar_2__[__temp_21__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__temp_20__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__temp_19__+(0-(__iter_2__+0))))]);
        //float __temp_23__;
        //__temp_23__ = (__temp_18__ + __temp_22__);
        //int __temp_24__;
        //__temp_24__ = __iter_6__;
        //int __temp_25__;
        //__temp_25__ = __iter_7__+(1);
        //int __temp_26__;
        //__temp_26__ = __iter_8__;
        //float __temp_27__;
        //__temp_27__ = (0.083000f * __tilevar_2__[__temp_26__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__temp_25__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__temp_24__+(0-(__iter_2__+0))))]);
        //float __temp_28__;
        //__temp_28__ = (__temp_23__ + __temp_27__);
        //int __temp_29__;
        //__temp_29__ = __iter_6__;
        //int __temp_30__;
        //__temp_30__ = __iter_7__+(-1);
        //int __temp_31__;
        //__temp_31__ = __iter_8__;
        //float __temp_32__;
        //__temp_32__ = (0.083000f * __tilevar_2__[__temp_31__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__temp_30__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__temp_29__+(0-(__iter_2__+0))))]);
        //float __temp_33__;
        //__temp_33__ = (__temp_28__ + __temp_32__);
        //int __temp_34__;
        //__temp_34__ = __iter_6__;
        //int __temp_35__;
        //__temp_35__ = __iter_7__+(-2);
        //int __temp_36__;
        //__temp_36__ = __iter_8__;
        //float __temp_37__;
        //__temp_37__ = (0.083000f * __tilevar_2__[__temp_36__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__temp_35__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__temp_34__+(0-(__iter_2__+0))))]);
        //float __temp_38__;
        //__temp_38__ = (__temp_33__ + __temp_37__);
        //int __temp_39__;
        //__temp_39__ = __iter_6__+(2);
        //int __temp_40__;
        //__temp_40__ = __iter_7__;
        //int __temp_41__;
        //__temp_41__ = __iter_8__;
        //float __temp_42__;
        //__temp_42__ = (0.083000f * __tilevar_2__[__temp_41__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__temp_40__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__temp_39__+(0-(__iter_2__+0))))]);
        //float __temp_43__;
        //__temp_43__ = (__temp_38__ + __temp_42__);
        //int __temp_44__;
        //__temp_44__ = __iter_6__+(1);
        //int __temp_45__;
        //__temp_45__ = __iter_7__;
        //int __temp_46__;
        //__temp_46__ = __iter_8__;
        //float __temp_47__;
        //__temp_47__ = (0.083000f * __tilevar_2__[__temp_46__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__temp_45__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__temp_44__+(0-(__iter_2__+0))))]);
        //float __temp_48__;
        //__temp_48__ = (__temp_43__ + __temp_47__);
        //int __temp_49__;
        //__temp_49__ = __iter_6__+(-1);
        //int __temp_50__;
        //__temp_50__ = __iter_7__;
        //int __temp_51__;
        //__temp_51__ = __iter_8__;
        //float __temp_52__;
        //__temp_52__ = (0.083000f * __tilevar_2__[__temp_51__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__temp_50__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__temp_49__+(0-(__iter_2__+0))))]);
        //float __temp_53__;
        //__temp_53__ = (__temp_48__ + __temp_52__);
        //int __temp_54__;
        //__temp_54__ = __iter_6__+(-2);
        //int __temp_55__;
        //__temp_55__ = __iter_7__;
        //int __temp_56__;
        //__temp_56__ = __iter_8__;
        //float __temp_57__;
        //__temp_57__ = (0.083000f * __tilevar_2__[__temp_56__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__temp_55__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__temp_54__+(0-(__iter_2__+0))))]);
        //float __temp_58__;
        //__temp_58__ = (__temp_53__ + __temp_57__);
        //int __temp_59__;
        //__temp_59__ = __iter_6__;
        //int __temp_60__;
        //__temp_60__ = __iter_7__;
        //int __temp_61__;
        //__temp_61__ = __iter_8__;
        //float __temp_62__;
        //__temp_62__ = (0.996000f * __tilevar_2__[__temp_61__+(0-(__iter_0__+0))+(FORMA_BLOCKDIM_X-0)*(__temp_60__+(0-(__iter_1__+0))+(FORMA_BLOCKDIM_Y-0)*(__temp_59__+(0-(__iter_2__+0))))]);
        //float __temp_63__;
        //__temp_63__ = (__temp_58__ - __temp_62__);
        //__tilevar_3__[__iter_8__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__iter_7__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__iter_6__+(0-(__iter_2__+2))))] = __temp_63__;
      }
    }
  }
  int __iter_9__;
  __iter_9__ = FORMA_MAX((__iter_2__+2),2) + (int)(threadIdx.z) ; 
  for( ; __iter_9__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-3),(L-3)) ; __iter_9__ += (int)(blockDim.z) ){
    int __iter_10__;
    __iter_10__ = FORMA_MAX((__iter_1__+2),2) + (int)(threadIdx.y) ; 
    if( __iter_10__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-3),(M-3)) ){
      int __iter_11__;
      __iter_11__ = FORMA_MAX((__iter_0__+2),2) + (int)(threadIdx.x) ; 
      if( __iter_11__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-3),(N-3)) ){
        if (__iter_9__ < (FORMA_MAX((__iter_2__+2),2)+4) || __iter_9__ > (FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-3),(L-3))-4) || __iter_10__ < (FORMA_MAX((__iter_1__+2),2)+4) || __iter_10__ > (FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-3),(M-3))-4) || __iter_11__ < (FORMA_MAX((__iter_0__+2),2)+4) || __iter_11__ > (FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-3),(N-3))-4)) {
          __copy_arr_0__[__iter_11__+(N-0)*(__iter_10__+(M-0)*(__iter_9__))] = __tilevar_3__[__iter_11__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__iter_10__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__iter_9__+(0-(__iter_2__+2))))];
        }
      }
    }
  }
  __syncthreads();
  int __iter_15__;
  __iter_15__ = FORMA_MAX((__iter_2__+4),2) + (int)(threadIdx.z) ; 
  for( ; __iter_15__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-5),(L-3)) ; __iter_15__ += (int)(blockDim.z) ){
    int __iter_16__;
    __iter_16__ = FORMA_MAX((__iter_1__+4),2) + (int)(threadIdx.y) ; 
    if( __iter_16__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-5),(M-3)) ){
      int __iter_17__;
      __iter_17__ = FORMA_MAX((__iter_0__+4),2) + (int)(threadIdx.x) ; 
      if( __iter_17__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-5),(N-3)) ){
        int __temp_89__;
        __temp_89__ = __iter_15__;
        int __temp_90__;
        __temp_90__ = __iter_16__;
        int __temp_91__;
        __temp_91__ = __iter_17__+(2);
        float __temp_92__;
        __temp_92__ = (0.083000f * __tilevar_3__[__temp_91__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__temp_90__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__temp_89__+(0-(__iter_2__+2))))]);
        int __temp_93__;
        __temp_93__ = __iter_15__;
        int __temp_94__;
        __temp_94__ = __iter_16__;
        int __temp_95__;
        __temp_95__ = __iter_17__+(1);
        float __temp_96__;
        __temp_96__ = (0.083000f * __tilevar_3__[__temp_95__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__temp_94__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__temp_93__+(0-(__iter_2__+2))))]);
        float __temp_97__;
        __temp_97__ = (__temp_92__ + __temp_96__);
        int __temp_98__;
        __temp_98__ = __iter_15__;
        int __temp_99__;
        __temp_99__ = __iter_16__;
        int __temp_100__;
        __temp_100__ = __iter_17__+(-1);
        float __temp_101__;
        __temp_101__ = (0.083000f * __tilevar_3__[__temp_100__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__temp_99__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__temp_98__+(0-(__iter_2__+2))))]);
        float __temp_102__;
        __temp_102__ = (__temp_97__ + __temp_101__);
        int __temp_103__;
        __temp_103__ = __iter_15__;
        int __temp_104__;
        __temp_104__ = __iter_16__;
        int __temp_105__;
        __temp_105__ = __iter_17__+(-2);
        float __temp_106__;
        __temp_106__ = (0.083000f * __tilevar_3__[__temp_105__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__temp_104__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__temp_103__+(0-(__iter_2__+2))))]);
        float __temp_107__;
        __temp_107__ = (__temp_102__ + __temp_106__);
        int __temp_108__;
        __temp_108__ = __iter_15__;
        int __temp_109__;
        __temp_109__ = __iter_16__+(2);
        int __temp_110__;
        __temp_110__ = __iter_17__;
        float __temp_111__;
        __temp_111__ = (0.083000f * __tilevar_3__[__temp_110__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__temp_109__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__temp_108__+(0-(__iter_2__+2))))]);
        float __temp_112__;
        __temp_112__ = (__temp_107__ + __temp_111__);
        int __temp_113__;
        __temp_113__ = __iter_15__;
        int __temp_114__;
        __temp_114__ = __iter_16__+(1);
        int __temp_115__;
        __temp_115__ = __iter_17__;
        float __temp_116__;
        __temp_116__ = (0.083000f * __tilevar_3__[__temp_115__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__temp_114__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__temp_113__+(0-(__iter_2__+2))))]);
        float __temp_117__;
        __temp_117__ = (__temp_112__ + __temp_116__);
        int __temp_118__;
        __temp_118__ = __iter_15__;
        int __temp_119__;
        __temp_119__ = __iter_16__+(-1);
        int __temp_120__;
        __temp_120__ = __iter_17__;
        float __temp_121__;
        __temp_121__ = (0.083000f * __tilevar_3__[__temp_120__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__temp_119__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__temp_118__+(0-(__iter_2__+2))))]);
        float __temp_122__;
        __temp_122__ = (__temp_117__ + __temp_121__);
        int __temp_123__;
        __temp_123__ = __iter_15__;
        int __temp_124__;
        __temp_124__ = __iter_16__+(-2);
        int __temp_125__;
        __temp_125__ = __iter_17__;
        float __temp_126__;
        __temp_126__ = (0.083000f * __tilevar_3__[__temp_125__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__temp_124__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__temp_123__+(0-(__iter_2__+2))))]);
        float __temp_127__;
        __temp_127__ = (__temp_122__ + __temp_126__);
        int __temp_128__;
        __temp_128__ = __iter_15__+(2);
        int __temp_129__;
        __temp_129__ = __iter_16__;
        int __temp_130__;
        __temp_130__ = __iter_17__;
        float __temp_131__;
        __temp_131__ = (0.083000f * __tilevar_3__[__temp_130__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__temp_129__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__temp_128__+(0-(__iter_2__+2))))]);
        float __temp_132__;
        __temp_132__ = (__temp_127__ + __temp_131__);
        int __temp_133__;
        __temp_133__ = __iter_15__+(1);
        int __temp_134__;
        __temp_134__ = __iter_16__;
        int __temp_135__;
        __temp_135__ = __iter_17__;
        float __temp_136__;
        __temp_136__ = (0.083000f * __tilevar_3__[__temp_135__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__temp_134__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__temp_133__+(0-(__iter_2__+2))))]);
        float __temp_137__;
        __temp_137__ = (__temp_132__ + __temp_136__);
        int __temp_138__;
        __temp_138__ = __iter_15__+(-1);
        int __temp_139__;
        __temp_139__ = __iter_16__;
        int __temp_140__;
        __temp_140__ = __iter_17__;
        float __temp_141__;
        __temp_141__ = (0.083000f * __tilevar_3__[__temp_140__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__temp_139__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__temp_138__+(0-(__iter_2__+2))))]);
        float __temp_142__;
        __temp_142__ = (__temp_137__ + __temp_141__);
        int __temp_143__;
        __temp_143__ = __iter_15__+(-2);
        int __temp_144__;
        __temp_144__ = __iter_16__;
        int __temp_145__;
        __temp_145__ = __iter_17__;
        float __temp_146__;
        __temp_146__ = (0.083000f * __tilevar_3__[__temp_145__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__temp_144__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__temp_143__+(0-(__iter_2__+2))))]);
        float __temp_147__;
        __temp_147__ = (__temp_142__ + __temp_146__);
        int __temp_148__;
        __temp_148__ = __iter_15__;
        int __temp_149__;
        __temp_149__ = __iter_16__;
        int __temp_150__;
        __temp_150__ = __iter_17__;
        float __temp_151__;
        __temp_151__ = (0.996000f * __tilevar_3__[__temp_150__+(0-(__iter_0__+2))+(FORMA_BLOCKDIM_X-0)*(__temp_149__+(0-(__iter_1__+2))+(FORMA_BLOCKDIM_Y-0)*(__temp_148__+(0-(__iter_2__+2))))]);
        float __temp_152__;
        __temp_152__ = (__temp_147__ - __temp_151__);
        __tilevar_4__[__iter_17__+(0-(__iter_0__+4))+(FORMA_BLOCKDIM_X-0)*(__iter_16__+(0-(__iter_1__+4))+(FORMA_BLOCKDIM_Y-0)*(__iter_15__+(0-(__iter_2__+4))))] = __temp_152__;
      }
    }
  }
  int __iter_18__;
  __iter_18__ = FORMA_MAX((__iter_2__+4),2) + (int)(threadIdx.z) ; 
  for( ; __iter_18__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-5),(L-3)) ; __iter_18__ += (int)(blockDim.z) ){
    int __iter_19__;
    __iter_19__ = FORMA_MAX((__iter_1__+4),2) + (int)(threadIdx.y) ; 
    if( __iter_19__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-5),(M-3)) ){
      int __iter_20__;
      __iter_20__ = FORMA_MAX((__iter_0__+4),2) + (int)(threadIdx.x) ; 
      if( __iter_20__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-5),(N-3)) ){
        if (__iter_18__ < (FORMA_MAX((__iter_2__+4),2)+4) || __iter_18__ > (FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-5),(L-3))-4) || __iter_19__ < (FORMA_MAX((__iter_1__+4),2)+4) || __iter_19__ > (FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-5),(M-3))-4) || __iter_20__ < (FORMA_MAX((__iter_0__+4),2)+4) || __iter_20__ > (FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-5),(N-3))-4)) {
          __copy_arr_1__[__iter_20__+(N-0)*(__iter_19__+(M-0)*(__iter_18__))] = __tilevar_4__[__iter_20__+(0-(__iter_0__+4))+(FORMA_BLOCKDIM_X-0)*(__iter_19__+(0-(__iter_1__+4))+(FORMA_BLOCKDIM_Y-0)*(__iter_18__+(0-(__iter_2__+4))))];
        }
      }
    }
  }
  __syncthreads();
  int __iter_24__;
  __iter_24__ = FORMA_MAX((__iter_2__+6),2) + (int)(threadIdx.z) ; 
  for( ; __iter_24__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-7),(L-3)) ; __iter_24__ += (int)(blockDim.z) ){
    int __iter_25__;
    __iter_25__ = FORMA_MAX((__iter_1__+6),2) + (int)(threadIdx.y) ; 
    if( __iter_25__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-7),(M-3)) ){
      int __iter_26__;
      __iter_26__ = FORMA_MAX((__iter_0__+6),2) + (int)(threadIdx.x) ; 
      if( __iter_26__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-7),(N-3)) ){
        float __temp_178__;
        __temp_178__ = (0.083000f * __tilevar_4__[__iter_26__+(2)+(0-(__iter_0__+4))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+4))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+4))))]);
        float __temp_179__;
        __temp_179__ = (0.083000f * __tilevar_4__[__iter_26__+(1)+(0-(__iter_0__+4))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+4))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+4))))]);
        float __temp_180__;
        __temp_180__ = (__temp_178__ + __temp_179__);
        float __temp_181__;
        __temp_181__ = (0.083000f * __tilevar_4__[__iter_26__+(-1)+(0-(__iter_0__+4))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+4))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+4))))]);
        float __temp_182__;
        __temp_182__ = (__temp_180__ + __temp_181__);
        float __temp_183__;
        __temp_183__ = (0.083000f * __tilevar_4__[__iter_26__+(-2)+(0-(__iter_0__+4))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+4))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+4))))]);
        float __temp_184__;
        __temp_184__ = (__temp_182__ + __temp_183__);
        float __temp_185__;
        __temp_185__ = (0.083000f * __tilevar_4__[__iter_26__+(0-(__iter_0__+4))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(2)+(0-(__iter_1__+4))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+4))))]);
        float __temp_186__;
        __temp_186__ = (__temp_184__ + __temp_185__);
        float __temp_187__;
        __temp_187__ = (0.083000f * __tilevar_4__[__iter_26__+(0-(__iter_0__+4))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(1)+(0-(__iter_1__+4))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+4))))]);
        float __temp_188__;
        __temp_188__ = (__temp_186__ + __temp_187__);
        float __temp_189__;
        __temp_189__ = (0.083000f * __tilevar_4__[__iter_26__+(0-(__iter_0__+4))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(-1)+(0-(__iter_1__+4))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+4))))]);
        float __temp_190__;
        __temp_190__ = (__temp_188__ + __temp_189__);
        float __temp_191__;
        __temp_191__ = (0.083000f * __tilevar_4__[__iter_26__+(0-(__iter_0__+4))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(-2)+(0-(__iter_1__+4))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+4))))]);
        float __temp_192__;
        __temp_192__ = (__temp_190__ + __temp_191__);
        float __temp_193__;
        __temp_193__ = (0.083000f * __tilevar_4__[__iter_26__+(0-(__iter_0__+4))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+4))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(2)+(0-(__iter_2__+4))))]);
        float __temp_194__;
        __temp_194__ = (__temp_192__ + __temp_193__);
        float __temp_195__;
        __temp_195__ = (0.083000f * __tilevar_4__[__iter_26__+(0-(__iter_0__+4))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+4))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(1)+(0-(__iter_2__+4))))]);
        float __temp_196__;
        __temp_196__ = (__temp_194__ + __temp_195__);
        float __temp_197__;
        __temp_197__ = (0.083000f * __tilevar_4__[__iter_26__+(0-(__iter_0__+4))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+4))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(-1)+(0-(__iter_2__+4))))]);
        float __temp_198__;
        __temp_198__ = (__temp_196__ + __temp_197__);
        float __temp_199__;
        __temp_199__ = (0.083000f * __tilevar_4__[__iter_26__+(0-(__iter_0__+4))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+4))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(-2)+(0-(__iter_2__+4))))]);
        float __temp_200__;
        __temp_200__ = (__temp_198__ + __temp_199__);
        float __temp_201__;
        __temp_201__ = (0.996000f * __tilevar_4__[__iter_26__+(0-(__iter_0__+4))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+4))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+4))))]);
        float __temp_202__;
        __temp_202__ = (__temp_200__ - __temp_201__);
        __tilevar_5__[__iter_26__+(0-(__iter_0__+6))+(FORMA_BLOCKDIM_X-0)*(__iter_25__+(0-(__iter_1__+6))+(FORMA_BLOCKDIM_Y-0)*(__iter_24__+(0-(__iter_2__+6))))] = __temp_202__;
      }
    }
  }
  int __iter_27__;
  __iter_27__ = FORMA_MAX((__iter_2__+6),2) + (int)(threadIdx.z) ; 
  for( ; __iter_27__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-7),(L-3)) ; __iter_27__ += (int)(blockDim.z) ){
    int __iter_28__;
    __iter_28__ = FORMA_MAX((__iter_1__+6),2) + (int)(threadIdx.y) ; 
    if( __iter_28__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-7),(M-3)) ){
      int __iter_29__;
      __iter_29__ = FORMA_MAX((__iter_0__+6),2) + (int)(threadIdx.x) ; 
      if( __iter_29__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-7),(N-3)) ){
        if (__iter_27__ < (FORMA_MAX((__iter_2__+6),2)+4) || __iter_27__ > (FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-7),(L-3))-4) || __iter_28__ < (FORMA_MAX((__iter_1__+6),2)+4) || __iter_28__ > (FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-7),(M-3))-4) || __iter_29__ < (FORMA_MAX((__iter_0__+6),2)+4) || __iter_29__ > (FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-7),(N-3))-4)) {
          __copy_arr_2__[__iter_29__+(N-0)*(__iter_28__+(M-0)*(__iter_27__))] = __tilevar_5__[__iter_29__+(0-(__iter_0__+6))+(FORMA_BLOCKDIM_X-0)*(__iter_28__+(0-(__iter_1__+6))+(FORMA_BLOCKDIM_Y-0)*(__iter_27__+(0-(__iter_2__+6))))];
        }
      }
    }
  }
  __syncthreads();
  int __iter_33__;
  __iter_33__ = FORMA_MAX((__iter_2__+8),2) + (int)(threadIdx.z) ; 
  for( ; __iter_33__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-9),(L-3)) ; __iter_33__ += (int)(blockDim.z) ){
    int __iter_34__;
    __iter_34__ = FORMA_MAX((__iter_1__+8),2) + (int)(threadIdx.y) ; 
    if( __iter_34__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-9),(M-3)) ){
      int __iter_35__;
      __iter_35__ = FORMA_MAX((__iter_0__+8),2) + (int)(threadIdx.x) ; 
      if( __iter_35__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-9),(N-3)) ){
        float __temp_228__;
        __temp_228__ = (0.083000f * __tilevar_5__[__iter_35__+(2)+(0-(__iter_0__+6))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+6))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+6))))]);
        float __temp_229__;
        __temp_229__ = (0.083000f * __tilevar_5__[__iter_35__+(1)+(0-(__iter_0__+6))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+6))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+6))))]);
        float __temp_230__;
        __temp_230__ = (__temp_228__ + __temp_229__);
        float __temp_231__;
        __temp_231__ = (0.083000f * __tilevar_5__[__iter_35__+(-1)+(0-(__iter_0__+6))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+6))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+6))))]);
        float __temp_232__;
        __temp_232__ = (__temp_230__ + __temp_231__);
        float __temp_233__;
        __temp_233__ = (0.083000f * __tilevar_5__[__iter_35__+(-2)+(0-(__iter_0__+6))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+6))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+6))))]);
        float __temp_234__;
        __temp_234__ = (__temp_232__ + __temp_233__);
        float __temp_235__;
        __temp_235__ = (0.083000f * __tilevar_5__[__iter_35__+(0-(__iter_0__+6))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(2)+(0-(__iter_1__+6))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+6))))]);
        float __temp_236__;
        __temp_236__ = (__temp_234__ + __temp_235__);
        float __temp_237__;
        __temp_237__ = (0.083000f * __tilevar_5__[__iter_35__+(0-(__iter_0__+6))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(1)+(0-(__iter_1__+6))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+6))))]);
        float __temp_238__;
        __temp_238__ = (__temp_236__ + __temp_237__);
        float __temp_239__;
        __temp_239__ = (0.083000f * __tilevar_5__[__iter_35__+(0-(__iter_0__+6))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(-1)+(0-(__iter_1__+6))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+6))))]);
        float __temp_240__;
        __temp_240__ = (__temp_238__ + __temp_239__);
        float __temp_241__;
        __temp_241__ = (0.083000f * __tilevar_5__[__iter_35__+(0-(__iter_0__+6))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(-2)+(0-(__iter_1__+6))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+6))))]);
        float __temp_242__;
        __temp_242__ = (__temp_240__ + __temp_241__);
        float __temp_243__;
        __temp_243__ = (0.083000f * __tilevar_5__[__iter_35__+(0-(__iter_0__+6))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+6))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(2)+(0-(__iter_2__+6))))]);
        float __temp_244__;
        __temp_244__ = (__temp_242__ + __temp_243__);
        float __temp_245__;
        __temp_245__ = (0.083000f * __tilevar_5__[__iter_35__+(0-(__iter_0__+6))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+6))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(1)+(0-(__iter_2__+6))))]);
        float __temp_246__;
        __temp_246__ = (__temp_244__ + __temp_245__);
        float __temp_247__;
        __temp_247__ = (0.083000f * __tilevar_5__[__iter_35__+(0-(__iter_0__+6))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+6))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(-1)+(0-(__iter_2__+6))))]);
        float __temp_248__;
        __temp_248__ = (__temp_246__ + __temp_247__);
        float __temp_249__;
        __temp_249__ = (0.083000f * __tilevar_5__[__iter_35__+(0-(__iter_0__+6))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+6))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(-2)+(0-(__iter_2__+6))))]);
        float __temp_250__;
        __temp_250__ = (__temp_248__ + __temp_249__);
        float __temp_251__;
        __temp_251__ = (0.996000f * __tilevar_5__[__iter_35__+(0-(__iter_0__+6))+(FORMA_BLOCKDIM_X-0)*(__iter_34__+(0-(__iter_1__+6))+(FORMA_BLOCKDIM_Y-0)*(__iter_33__+(0-(__iter_2__+6))))]);
        float __temp_252__;
        __temp_252__ = (__temp_250__ - __temp_251__);
        __var_1__[__iter_35__+(N-0)*(__iter_34__+(M-0)*(__iter_33__))] = __temp_252__;
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

__global__ void __kernel___forma_kernel__1__(float * __restrict__ input, int L, int M, int N, int FORMA_BLOCKDIM_X, int FORMA_BLOCKDIM_Y, int FORMA_BLOCKDIM_Z, float * __restrict__ __copy_arr_0__){
  int __iter_0__;
  __iter_0__ = (int)(blockIdx.x)*(int)(FORMA_BLOCKDIM_X);
  int __iter_1__;
  __iter_1__ = (int)(blockIdx.y)*(int)(FORMA_BLOCKDIM_Y);
  int __iter_2__;
  __iter_2__ = (int)(blockIdx.z)*(int)(FORMA_BLOCKDIM_Z);
  int __iter_12__;
  __iter_12__ = FORMA_MAX(__iter_2__,2) + (int)(threadIdx.z) ; 
  for( ; __iter_12__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-1),(L-3)) ; __iter_12__ += (int)(blockDim.z) ){
    int __iter_13__;
    __iter_13__ = FORMA_MAX(__iter_1__,2) + (int)(threadIdx.y) ; 
    for( ; __iter_13__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-1),(M-3)) ; __iter_13__ += (int)(blockDim.y) ){
      int __iter_14__;
      __iter_14__ = FORMA_MAX(__iter_0__,2) + (int)(threadIdx.x) ; 
      if( __iter_14__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-1),(N-3)) ){
        if (__iter_12__ < FORMA_MAX((__iter_2__+2),2) || __iter_12__ > FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-3),(L-3)) || __iter_13__ < FORMA_MAX((__iter_1__+2),2) || __iter_13__ > FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-3),(M-3)) || __iter_14__ < FORMA_MAX((__iter_0__+2),2) || __iter_14__ > FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-3),(N-3))) {
          float __temp_64__;
          __temp_64__ = (0.083000f * input[__iter_14__+(2)+(N-0)*(__iter_13__+(M-0)*(__iter_12__))]);
          float __temp_65__;
          __temp_65__ = (0.083000f * input[__iter_14__+(1)+(N-0)*(__iter_13__+(M-0)*(__iter_12__))]);
          float __temp_66__;
          __temp_66__ = (__temp_64__ + __temp_65__);
          float __temp_67__;
          __temp_67__ = (0.083000f * input[__iter_14__+(-1)+(N-0)*(__iter_13__+(M-0)*(__iter_12__))]);
          float __temp_68__;
          __temp_68__ = (__temp_66__ + __temp_67__);
          float __temp_69__;
          __temp_69__ = (0.083000f * input[__iter_14__+(-2)+(N-0)*(__iter_13__+(M-0)*(__iter_12__))]);
          float __temp_70__;
          __temp_70__ = (__temp_68__ + __temp_69__);
          float __temp_71__;
          __temp_71__ = (0.083000f * input[__iter_14__+(N-0)*(__iter_13__+(2)+(M-0)*(__iter_12__))]);
          float __temp_72__;
          __temp_72__ = (__temp_70__ + __temp_71__);
          float __temp_73__;
          __temp_73__ = (0.083000f * input[__iter_14__+(N-0)*(__iter_13__+(1)+(M-0)*(__iter_12__))]);
          float __temp_74__;
          __temp_74__ = (__temp_72__ + __temp_73__);
          float __temp_75__;
          __temp_75__ = (0.083000f * input[__iter_14__+(N-0)*(__iter_13__+(-1)+(M-0)*(__iter_12__))]);
          float __temp_76__;
          __temp_76__ = (__temp_74__ + __temp_75__);
          float __temp_77__;
          __temp_77__ = (0.083000f * input[__iter_14__+(N-0)*(__iter_13__+(-2)+(M-0)*(__iter_12__))]);
          float __temp_78__;
          __temp_78__ = (__temp_76__ + __temp_77__);
          float __temp_79__;
          __temp_79__ = (0.083000f * input[__iter_14__+(N-0)*(__iter_13__+(M-0)*(__iter_12__+(2)))]);
          float __temp_80__;
          __temp_80__ = (__temp_78__ + __temp_79__);
          float __temp_81__;
          __temp_81__ = (0.083000f * input[__iter_14__+(N-0)*(__iter_13__+(M-0)*(__iter_12__+(1)))]);
          float __temp_82__;
          __temp_82__ = (__temp_80__ + __temp_81__);
          float __temp_83__;
          __temp_83__ = (0.083000f * input[__iter_14__+(N-0)*(__iter_13__+(M-0)*(__iter_12__+(-1)))]);
          float __temp_84__;
          __temp_84__ = (__temp_82__ + __temp_83__);
          float __temp_85__;
          __temp_85__ = (0.083000f * input[__iter_14__+(N-0)*(__iter_13__+(M-0)*(__iter_12__+(-2)))]);
          float __temp_86__;
          __temp_86__ = (__temp_84__ + __temp_85__);
          float __temp_87__;
          __temp_87__ = (0.996000f * input[__iter_14__+(N-0)*(__iter_13__+(M-0)*(__iter_12__))]);
          float __temp_88__;
          __temp_88__ = (__temp_86__ - __temp_87__);
          __copy_arr_0__[__iter_14__+(N-0)*(__iter_13__+(M-0)*(__iter_12__))] = __temp_88__;
        }
      }
    }
  }
}

__global__ void __kernel___forma_kernel__2__(float * __restrict__ __copy_arr_0__, int L, int M, int N, int FORMA_BLOCKDIM_X, int FORMA_BLOCKDIM_Y, int FORMA_BLOCKDIM_Z, float * __restrict__ __copy_arr_1__){
  int __iter_0__;
  __iter_0__ = (int)(blockIdx.x)*(int)(FORMA_BLOCKDIM_X);
  int __iter_1__;
  __iter_1__ = (int)(blockIdx.y)*(int)(FORMA_BLOCKDIM_Y);
  int __iter_2__;
  __iter_2__ = (int)(blockIdx.z)*(int)(FORMA_BLOCKDIM_Z);
  int __iter_21__;
  __iter_21__ = FORMA_MAX(__iter_2__,2) + (int)(threadIdx.z) ; 
  for( ; __iter_21__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-1),(L-3)) ; __iter_21__ += (int)(blockDim.z) ){
    int __iter_22__;
    __iter_22__ = FORMA_MAX(__iter_1__,2) + (int)(threadIdx.y) ; 
    for( ; __iter_22__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-1),(M-3)) ; __iter_22__ += (int)(blockDim.y) ){
      int __iter_23__;
      __iter_23__ = FORMA_MAX(__iter_0__,2) + (int)(threadIdx.x) ; 
      if( __iter_23__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-1),(N-3)) ){
        if (__iter_21__ < FORMA_MAX((__iter_2__+4),2) || __iter_21__ > FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-5),(L-3)) || __iter_22__ < FORMA_MAX((__iter_1__+4),2) || __iter_22__ > FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-5),(M-3)) || __iter_23__ < FORMA_MAX((__iter_0__+4),2) || __iter_23__ > FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-5),(N-3))) {
          float __temp_153__;
          __temp_153__ = (0.083000f * __copy_arr_0__[__iter_23__+(2)+(N-0)*(__iter_22__+(M-0)*(__iter_21__))]);
          float __temp_154__;
          __temp_154__ = (0.083000f * __copy_arr_0__[__iter_23__+(1)+(N-0)*(__iter_22__+(M-0)*(__iter_21__))]);
          float __temp_155__;
          __temp_155__ = (__temp_153__ + __temp_154__);
          float __temp_156__;
          __temp_156__ = (0.083000f * __copy_arr_0__[__iter_23__+(-1)+(N-0)*(__iter_22__+(M-0)*(__iter_21__))]);
          float __temp_157__;
          __temp_157__ = (__temp_155__ + __temp_156__);
          float __temp_158__;
          __temp_158__ = (0.083000f * __copy_arr_0__[__iter_23__+(-2)+(N-0)*(__iter_22__+(M-0)*(__iter_21__))]);
          float __temp_159__;
          __temp_159__ = (__temp_157__ + __temp_158__);
          float __temp_160__;
          __temp_160__ = (0.083000f * __copy_arr_0__[__iter_23__+(N-0)*(__iter_22__+(2)+(M-0)*(__iter_21__))]);
          float __temp_161__;
          __temp_161__ = (__temp_159__ + __temp_160__);
          float __temp_162__;
          __temp_162__ = (0.083000f * __copy_arr_0__[__iter_23__+(N-0)*(__iter_22__+(1)+(M-0)*(__iter_21__))]);
          float __temp_163__;
          __temp_163__ = (__temp_161__ + __temp_162__);
          float __temp_164__;
          __temp_164__ = (0.083000f * __copy_arr_0__[__iter_23__+(N-0)*(__iter_22__+(-1)+(M-0)*(__iter_21__))]);
          float __temp_165__;
          __temp_165__ = (__temp_163__ + __temp_164__);
          float __temp_166__;
          __temp_166__ = (0.083000f * __copy_arr_0__[__iter_23__+(N-0)*(__iter_22__+(-2)+(M-0)*(__iter_21__))]);
          float __temp_167__;
          __temp_167__ = (__temp_165__ + __temp_166__);
          float __temp_168__;
          __temp_168__ = (0.083000f * __copy_arr_0__[__iter_23__+(N-0)*(__iter_22__+(M-0)*(__iter_21__+(2)))]);
          float __temp_169__;
          __temp_169__ = (__temp_167__ + __temp_168__);
          float __temp_170__;
          __temp_170__ = (0.083000f * __copy_arr_0__[__iter_23__+(N-0)*(__iter_22__+(M-0)*(__iter_21__+(1)))]);
          float __temp_171__;
          __temp_171__ = (__temp_169__ + __temp_170__);
          float __temp_172__;
          __temp_172__ = (0.083000f * __copy_arr_0__[__iter_23__+(N-0)*(__iter_22__+(M-0)*(__iter_21__+(-1)))]);
          float __temp_173__;
          __temp_173__ = (__temp_171__ + __temp_172__);
          float __temp_174__;
          __temp_174__ = (0.083000f * __copy_arr_0__[__iter_23__+(N-0)*(__iter_22__+(M-0)*(__iter_21__+(-2)))]);
          float __temp_175__;
          __temp_175__ = (__temp_173__ + __temp_174__);
          float __temp_176__;
          __temp_176__ = (0.996000f * __copy_arr_0__[__iter_23__+(N-0)*(__iter_22__+(M-0)*(__iter_21__))]);
          float __temp_177__;
          __temp_177__ = (__temp_175__ - __temp_176__);
          __copy_arr_1__[__iter_23__+(N-0)*(__iter_22__+(M-0)*(__iter_21__))] = __temp_177__;
        }
      }
    }
  }
}

__global__ void __kernel___forma_kernel__3__(float * __restrict__ __copy_arr_1__, int L, int M, int N, int FORMA_BLOCKDIM_X, int FORMA_BLOCKDIM_Y, int FORMA_BLOCKDIM_Z, float * __restrict__ __copy_arr_2__){
  int __iter_0__;
  __iter_0__ = (int)(blockIdx.x)*(int)(FORMA_BLOCKDIM_X);
  int __iter_1__;
  __iter_1__ = (int)(blockIdx.y)*(int)(FORMA_BLOCKDIM_Y);
  int __iter_2__;
  __iter_2__ = (int)(blockIdx.z)*(int)(FORMA_BLOCKDIM_Z);
  int __iter_30__;
  __iter_30__ = FORMA_MAX(__iter_2__,2) + (int)(threadIdx.z) ; 
  for( ; __iter_30__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-1),(L-3)) ; __iter_30__ += (int)(blockDim.z) ){
    int __iter_31__;
    __iter_31__ = FORMA_MAX(__iter_1__,2) + (int)(threadIdx.y) ; 
    for( ; __iter_31__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-1),(M-3)) ; __iter_31__ += (int)(blockDim.y) ){
      int __iter_32__;
      __iter_32__ = FORMA_MAX(__iter_0__,2) + (int)(threadIdx.x) ; 
      if( __iter_32__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-1),(N-3)) ){
        if (__iter_30__ < FORMA_MAX((__iter_2__+6),2) || __iter_30__ > FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-7),(L-3)) || __iter_31__ < FORMA_MAX((__iter_1__+6),2) || __iter_31__ > FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-7),(M-3)) || __iter_32__ < FORMA_MAX((__iter_0__+6),2) || __iter_32__ > FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-7),(N-3))) {
          float __temp_203__;
          __temp_203__ = (0.083000f * __copy_arr_1__[__iter_32__+(2)+(N-0)*(__iter_31__+(M-0)*(__iter_30__))]);
          float __temp_204__;
          __temp_204__ = (0.083000f * __copy_arr_1__[__iter_32__+(1)+(N-0)*(__iter_31__+(M-0)*(__iter_30__))]);
          float __temp_205__;
          __temp_205__ = (__temp_203__ + __temp_204__);
          float __temp_206__;
          __temp_206__ = (0.083000f * __copy_arr_1__[__iter_32__+(-1)+(N-0)*(__iter_31__+(M-0)*(__iter_30__))]);
          float __temp_207__;
          __temp_207__ = (__temp_205__ + __temp_206__);
          float __temp_208__;
          __temp_208__ = (0.083000f * __copy_arr_1__[__iter_32__+(-2)+(N-0)*(__iter_31__+(M-0)*(__iter_30__))]);
          float __temp_209__;
          __temp_209__ = (__temp_207__ + __temp_208__);
          float __temp_210__;
          __temp_210__ = (0.083000f * __copy_arr_1__[__iter_32__+(N-0)*(__iter_31__+(2)+(M-0)*(__iter_30__))]);
          float __temp_211__;
          __temp_211__ = (__temp_209__ + __temp_210__);
          float __temp_212__;
          __temp_212__ = (0.083000f * __copy_arr_1__[__iter_32__+(N-0)*(__iter_31__+(1)+(M-0)*(__iter_30__))]);
          float __temp_213__;
          __temp_213__ = (__temp_211__ + __temp_212__);
          float __temp_214__;
          __temp_214__ = (0.083000f * __copy_arr_1__[__iter_32__+(N-0)*(__iter_31__+(-1)+(M-0)*(__iter_30__))]);
          float __temp_215__;
          __temp_215__ = (__temp_213__ + __temp_214__);
          float __temp_216__;
          __temp_216__ = (0.083000f * __copy_arr_1__[__iter_32__+(N-0)*(__iter_31__+(-2)+(M-0)*(__iter_30__))]);
          float __temp_217__;
          __temp_217__ = (__temp_215__ + __temp_216__);
          float __temp_218__;
          __temp_218__ = (0.083000f * __copy_arr_1__[__iter_32__+(N-0)*(__iter_31__+(M-0)*(__iter_30__+(2)))]);
          float __temp_219__;
          __temp_219__ = (__temp_217__ + __temp_218__);
          float __temp_220__;
          __temp_220__ = (0.083000f * __copy_arr_1__[__iter_32__+(N-0)*(__iter_31__+(M-0)*(__iter_30__+(1)))]);
          float __temp_221__;
          __temp_221__ = (__temp_219__ + __temp_220__);
          float __temp_222__;
          __temp_222__ = (0.083000f * __copy_arr_1__[__iter_32__+(N-0)*(__iter_31__+(M-0)*(__iter_30__+(-1)))]);
          float __temp_223__;
          __temp_223__ = (__temp_221__ + __temp_222__);
          float __temp_224__;
          __temp_224__ = (0.083000f * __copy_arr_1__[__iter_32__+(N-0)*(__iter_31__+(M-0)*(__iter_30__+(-2)))]);
          float __temp_225__;
          __temp_225__ = (__temp_223__ + __temp_224__);
          float __temp_226__;
          __temp_226__ = (0.996000f * __copy_arr_1__[__iter_32__+(N-0)*(__iter_31__+(M-0)*(__iter_30__))]);
          float __temp_227__;
          __temp_227__ = (__temp_225__ - __temp_226__);
          __copy_arr_2__[__iter_32__+(N-0)*(__iter_31__+(M-0)*(__iter_30__))] = __temp_227__;
        }
      }
    }
  }
}

__global__ void __kernel___forma_kernel__4__(float * __restrict__ __copy_arr_2__, int L, int M, int N, int FORMA_BLOCKDIM_X, int FORMA_BLOCKDIM_Y, int FORMA_BLOCKDIM_Z, float * __restrict__ __var_1__){
  int __iter_0__;
  __iter_0__ = (int)(blockIdx.x)*(int)(FORMA_BLOCKDIM_X);
  int __iter_1__;
  __iter_1__ = (int)(blockIdx.y)*(int)(FORMA_BLOCKDIM_Y);
  int __iter_2__;
  __iter_2__ = (int)(blockIdx.z)*(int)(FORMA_BLOCKDIM_Z);
  int __iter_36__;
  __iter_36__ = FORMA_MAX(__iter_2__,2) + (int)(threadIdx.z) ; 
  for( ; __iter_36__ <= FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-1),(L-3)) ; __iter_36__ += (int)(blockDim.z) ){
    int __iter_37__;
    __iter_37__ = FORMA_MAX(__iter_1__,2) + (int)(threadIdx.y) ; 
    for( ; __iter_37__ <= FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-1),(M-3)) ; __iter_37__ += (int)(blockDim.y) ){
      int __iter_38__;
      __iter_38__ = FORMA_MAX(__iter_0__,2) + (int)(threadIdx.x) ; 
      if( __iter_38__ <= FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-1),(N-3)) ){
        if (__iter_36__ < FORMA_MAX((__iter_2__+8),2) || __iter_36__ > FORMA_MIN(((__iter_2__+FORMA_BLOCKDIM_Z)-9),(L-3)) || __iter_37__ < FORMA_MAX((__iter_1__+8),2) || __iter_37__ > FORMA_MIN(((__iter_1__+FORMA_BLOCKDIM_Y)-9),(M-3)) || __iter_38__ < FORMA_MAX((__iter_0__+8),2) || __iter_38__ > FORMA_MIN(((__iter_0__+FORMA_BLOCKDIM_X)-9),(N-3))) {
          float __temp_253__;
          __temp_253__ = (0.083000f * __copy_arr_2__[__iter_38__+(2)+(N-0)*(__iter_37__+(M-0)*(__iter_36__))]);
          float __temp_254__;
          __temp_254__ = (0.083000f * __copy_arr_2__[__iter_38__+(1)+(N-0)*(__iter_37__+(M-0)*(__iter_36__))]);
          float __temp_255__;
          __temp_255__ = (__temp_253__ + __temp_254__);
          float __temp_256__;
          __temp_256__ = (0.083000f * __copy_arr_2__[__iter_38__+(-1)+(N-0)*(__iter_37__+(M-0)*(__iter_36__))]);
          float __temp_257__;
          __temp_257__ = (__temp_255__ + __temp_256__);
          float __temp_258__;
          __temp_258__ = (0.083000f * __copy_arr_2__[__iter_38__+(-2)+(N-0)*(__iter_37__+(M-0)*(__iter_36__))]);
          float __temp_259__;
          __temp_259__ = (__temp_257__ + __temp_258__);
          float __temp_260__;
          __temp_260__ = (0.083000f * __copy_arr_2__[__iter_38__+(N-0)*(__iter_37__+(2)+(M-0)*(__iter_36__))]);
          float __temp_261__;
          __temp_261__ = (__temp_259__ + __temp_260__);
          float __temp_262__;
          __temp_262__ = (0.083000f * __copy_arr_2__[__iter_38__+(N-0)*(__iter_37__+(1)+(M-0)*(__iter_36__))]);
          float __temp_263__;
          __temp_263__ = (__temp_261__ + __temp_262__);
          float __temp_264__;
          __temp_264__ = (0.083000f * __copy_arr_2__[__iter_38__+(N-0)*(__iter_37__+(-1)+(M-0)*(__iter_36__))]);
          float __temp_265__;
          __temp_265__ = (__temp_263__ + __temp_264__);
          float __temp_266__;
          __temp_266__ = (0.083000f * __copy_arr_2__[__iter_38__+(N-0)*(__iter_37__+(-2)+(M-0)*(__iter_36__))]);
          float __temp_267__;
          __temp_267__ = (__temp_265__ + __temp_266__);
          float __temp_268__;
          __temp_268__ = (0.083000f * __copy_arr_2__[__iter_38__+(N-0)*(__iter_37__+(M-0)*(__iter_36__+(2)))]);
          float __temp_269__;
          __temp_269__ = (__temp_267__ + __temp_268__);
          float __temp_270__;
          __temp_270__ = (0.083000f * __copy_arr_2__[__iter_38__+(N-0)*(__iter_37__+(M-0)*(__iter_36__+(1)))]);
          float __temp_271__;
          __temp_271__ = (__temp_269__ + __temp_270__);
          float __temp_272__;
          __temp_272__ = (0.083000f * __copy_arr_2__[__iter_38__+(N-0)*(__iter_37__+(M-0)*(__iter_36__+(-1)))]);
          float __temp_273__;
          __temp_273__ = (__temp_271__ + __temp_272__);
          float __temp_274__;
          __temp_274__ = (0.083000f * __copy_arr_2__[__iter_38__+(N-0)*(__iter_37__+(M-0)*(__iter_36__+(-2)))]);
          float __temp_275__;
          __temp_275__ = (__temp_273__ + __temp_274__);
          float __temp_276__;
          __temp_276__ = (0.996000f * __copy_arr_2__[__iter_38__+(N-0)*(__iter_37__+(M-0)*(__iter_36__))]);
          float __temp_277__;
          __temp_277__ = (__temp_275__ - __temp_276__);
          __var_1__[__iter_38__+(N-0)*(__iter_37__+(M-0)*(__iter_36__))] = __temp_277__;
        }
      }
    }
  }
}

/*Device code End */
/* Host Code Begin */
extern "C" void j3d13pt(float * h_input, int L, int M, int N, float * __var_0__){

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
  int __max_occupancy_blocksize___kernel___forma_kernel__0__;
  int _max_occupancy_gridsize___kernel___forma_kernel__0__;
  cudaOccupancyMaxPotentialBlockSize(&_max_occupancy_gridsize___kernel___forma_kernel__0__,&__max_occupancy_blocksize___kernel___forma_kernel__0__,(const void*)__kernel___forma_kernel__0__,0,0);
  int __max_occupancy_blocksize___kernel___forma_kernel__0___0 = pow((double)__max_occupancy_blocksize___kernel___forma_kernel__0__, (double)(1.0/(double)3));
  __max_occupancy_blocksize___kernel___forma_kernel__0___0 = FORMA_MAX(__max_occupancy_blocksize___kernel___forma_kernel__0___0/32, 1)*32;
  int __block_0___kernel___forma_kernel__0__ = FORMA_MAX(FORMA_MIN(FORMA_MIN(__max_occupancy_blocksize___kernel___forma_kernel__0___0,FORMA_MAX((__size_0___kernel___forma_kernel__0__)/32,1)*32),FORMA_MAX_BLOCKDIM_0),13);
  __max_occupancy_blocksize___kernel___forma_kernel__0__ /= __block_0___kernel___forma_kernel__0__;
  int __max_occupancy_blocksize___kernel___forma_kernel__0___1 = pow((double)__max_occupancy_blocksize___kernel___forma_kernel__0__, (double)(1.0/(double)2));
  int __block_1___kernel___forma_kernel__0__ = FORMA_MAX(FORMA_MIN(FORMA_MIN(__max_occupancy_blocksize___kernel___forma_kernel__0___1,__size_1___kernel___forma_kernel__0__),FORMA_MAX_BLOCKDIM_1),13);
  __max_occupancy_blocksize___kernel___forma_kernel__0__ /= __block_1___kernel___forma_kernel__0__;
  int __max_occupancy_blocksize___kernel___forma_kernel__0___2 = __max_occupancy_blocksize___kernel___forma_kernel__0__;
  int __block_2___kernel___forma_kernel__0__ = FORMA_MAX(FORMA_MIN(FORMA_MIN(__max_occupancy_blocksize___kernel___forma_kernel__0___2,__size_2___kernel___forma_kernel__0__),FORMA_MAX_BLOCKDIM_2),13);
  __max_occupancy_blocksize___kernel___forma_kernel__0__ /= __block_2___kernel___forma_kernel__0__;
  dim3 __blockConfig___kernel___forma_kernel__0__(__block_0___kernel___forma_kernel__0__,__block_1___kernel___forma_kernel__0__,__block_2___kernel___forma_kernel__0__);
  int __SMemSize___kernel___forma_kernel__0__ = 0;
  __SMemSize___kernel___forma_kernel__0__ = __blockSizeToSMemSize___kernel___forma_kernel__0__(__blockConfig___kernel___forma_kernel__0__);
  while( __SMemSize___kernel___forma_kernel__0__ > __FORMA_MAX_SHARED_MEM__){
    if( __blockConfig___kernel___forma_kernel__0__.z/2 > 13)
      __blockConfig___kernel___forma_kernel__0__.z /= 2;
    __SMemSize___kernel___forma_kernel__0__ = __blockSizeToSMemSize___kernel___forma_kernel__0__(__blockConfig___kernel___forma_kernel__0__);
    if( __SMemSize___kernel___forma_kernel__0__ <= __FORMA_MAX_SHARED_MEM__)
      break;
    if( __blockConfig___kernel___forma_kernel__0__.y/2 > 13)
      __blockConfig___kernel___forma_kernel__0__.y /= 2;
    __SMemSize___kernel___forma_kernel__0__ = __blockSizeToSMemSize___kernel___forma_kernel__0__(__blockConfig___kernel___forma_kernel__0__);
    if( __SMemSize___kernel___forma_kernel__0__ <= __FORMA_MAX_SHARED_MEM__)
      break;
    if( __blockConfig___kernel___forma_kernel__0__.x/2 > FORMA_MIN(32,13))
      __blockConfig___kernel___forma_kernel__0__.x /= 2;
    __SMemSize___kernel___forma_kernel__0__ = __blockSizeToSMemSize___kernel___forma_kernel__0__(__blockConfig___kernel___forma_kernel__0__);
  }
  int __grid_0___kernel___forma_kernel__0__ = FORMA_CEIL(__size_0___kernel___forma_kernel__0__,__blockConfig___kernel___forma_kernel__0__.x);
  int __grid_1___kernel___forma_kernel__0__ = FORMA_CEIL(__size_1___kernel___forma_kernel__0__,__blockConfig___kernel___forma_kernel__0__.y);
  int __grid_2___kernel___forma_kernel__0__ = FORMA_CEIL(__size_2___kernel___forma_kernel__0__,__blockConfig___kernel___forma_kernel__0__.z);
  dim3 __gridConfig___kernel___forma_kernel__0__(__grid_0___kernel___forma_kernel__0__,__grid_1___kernel___forma_kernel__0__,__grid_2___kernel___forma_kernel__0__);
  dim3 __blockConfig_st__kernel___forma_kernel__0__(__blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y, 2);
  __kernel___forma_kernel__0__<<<__gridConfig___kernel___forma_kernel__0__, __blockConfig_st__kernel___forma_kernel__0__, __SMemSize___kernel___forma_kernel__0__>>> (input, L, M, N, __copy_arr_0__, __copy_arr_1__, __copy_arr_2__, __blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y, __blockConfig___kernel___forma_kernel__0__.z, __var_1__);
  Check_CUDA_Error("Kernel Launch Error!! : __kernel___forma_kernel__0__\n");
  dim3 __blockConfig___kernel___forma_kernel__1__(__blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y/2, 1);
  __kernel___forma_kernel__1__<<<__gridConfig___kernel___forma_kernel__0__, __blockConfig___kernel___forma_kernel__1__>>> (input, L, M, N, __blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y, __blockConfig___kernel___forma_kernel__0__.z, __copy_arr_0__);
  Check_CUDA_Error("Kernel Launch Error!! : __kernel___forma_kernel__1__\n");

  dim3 __blockConfig___kernel___forma_kernel__2__(__blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y/2, 1);
  __kernel___forma_kernel__2__<<<__gridConfig___kernel___forma_kernel__0__, __blockConfig___kernel___forma_kernel__2__>>> (__copy_arr_0__, L, M, N, __blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y, __blockConfig___kernel___forma_kernel__0__.z, __copy_arr_1__);
  Check_CUDA_Error("Kernel Launch Error!! : __kernel___forma_kernel__2__\n");

  dim3 __blockConfig___kernel___forma_kernel__3__(__blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y/2, 1);
  __kernel___forma_kernel__3__<<<__gridConfig___kernel___forma_kernel__0__, __blockConfig___kernel___forma_kernel__3__>>> (__copy_arr_1__, L, M, N, __blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y, __blockConfig___kernel___forma_kernel__0__.z, __copy_arr_2__);
  Check_CUDA_Error("Kernel Launch Error!! : __kernel___forma_kernel__3__\n");

  dim3 __blockConfig___kernel___forma_kernel__4__(__blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y/2, 1);
  __kernel___forma_kernel__4__<<<__gridConfig___kernel___forma_kernel__0__, __blockConfig___kernel___forma_kernel__4__>>> (__copy_arr_2__, L, M, N, __blockConfig___kernel___forma_kernel__0__.x, __blockConfig___kernel___forma_kernel__0__.y, __blockConfig___kernel___forma_kernel__0__.z, __var_1__);
  Check_CUDA_Error("Kernel Launch Error!! : __kernel___forma_kernel__4__\n");

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
