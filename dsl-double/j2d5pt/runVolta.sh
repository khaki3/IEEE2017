nvcc -O3 -maxrregcount=32 -ccbin=g++ -std=c++11 -Xcompiler "-fPIC -fopenmp -O3 -fno-strict-aliasing" --use_fast_math -Xptxas "-v -dlcm=cg" -gencode arch=compute_60,code=sm_60 ../../cuda_header.cu ../jacobi_gold.cpp jacobi.driver.cpp test-stream-y-overlap-4-reg-divide-y.cu -o test
nvprof --print-gpu-trace ./test
