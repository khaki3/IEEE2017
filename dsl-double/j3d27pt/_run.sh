for regCnt in -maxrregcount=32 -maxrregcount=64 ""; do
    for f in test-stream-z-overlap-4-opt.cu; do
	echo Running $f for regCnt $regCnt
	nvcc -D _TIMER_ -O3 $regCnt -ccbin=g++ -std=c++11 -Xcompiler -fopenmp --use_fast_math -Xptxas "-v" -gencode arch=compute_70,code=sm_70 -gencode arch=compute_60,code=sm_60 ../common/cuda_header.cu ./*_gold.cpp *.driver.cpp $f -o test 2>&1 | egrep 'reg|sm|spill';
	./test 514 514 514
    done
done

