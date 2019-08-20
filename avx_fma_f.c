// copyright colfaxresearch.com 2017 https://colfaxresearch.com/skl-avx512/
// icpc -qopenmp -xHost -S  avx_fma.c && time ./avx_fma
// -qopt-report=5 -qopt-report-phase=vec -qopt-report-embed
// export KMP_HW_SUBSET=1t # to disable hyperthreading
// export KMP_AFFINITY=compact,1 # core affinity
// (gcc -O3 -march=skylake -c avx_fma_f.c -o avx_fma_f.gcc.c -fopt-info-vec-all)
// results for: intel i7 skylake 6700k 4.1 Ghz
// look-up /proc/cpuinfo -> model=94 in hex is 0x5E -> intel 64 Architecture Optimization Reference Manual 
// for instruction latencies p.747 Appendix C.3 (https://software.intel.com/en-us/articles/intel-sdm#combined)
// skylake SP has it's own pdf with yuuge latency/troughput table
// export KMP_HW_SUBSET=2t vs. export KMP_HW_SUBSET=1t // 8 vs 4 threads; 4 cores SMT on/off
// 2 chained_fmas -> 250 gflops     @4 130
// 3 chained_fmas -> 350 gflops
// 4 chained         500            @4 260    note: 8 threads ~= 8 chained_fmas with 4 threads 520 gflops
// 5                 510            @4 320
// 8                 520            @4 520
// 10                510
// 12                515
// 14                510            @ 4 523.5 (MAXIMUM)
// 15                480            @ 4 threads = 380 gflops (register spilling starts)
// 16                420            @ 4 255
// 17                500            @ 4 420
// 18                420            @ 4 290
// 20                400            @ 4 311
// 24                400            @ 4 322

//icc -g -qopenmp -xHost -qopt-report=5 -qopt-report-phase=vec -qopt-report-embed  avx_fma_f.c -o avx_fma_float_2_8threads_v2
// while :;do ./avx_fma_float_2_8threads_v2 ; done

#include <stdio.h>
#include <omp.h>

#define VECTOR_WIDTH 8
const long  n_trials=30000000000L;
const int flops_per_calc =2;
const int n_chained_fmas        = 8 ;

int main(){
#pragma omp parallel
{ } // warm-up threads
//omp_set_num_threads(4);
const double t0=omp_get_wtime();

#pragma omp parallel
{ 
    float x __attribute__((aligned(64)));

    float fa[VECTOR_WIDTH*n_chained_fmas] __attribute__((aligned(64)));
    float fb[VECTOR_WIDTH] __attribute__((aligned(64)));
    float fc[VECTOR_WIDTH] __attribute__((aligned(64)));
    fa[0:VECTOR_WIDTH*n_chained_fmas]=0.0;
    fb[0:VECTOR_WIDTH]=0.5;
    fc[0:VECTOR_WIDTH]=1.0;
    register float *fa01 = fa + 0*VECTOR_WIDTH;
    register float *fa02 = fa + 1*VECTOR_WIDTH;
    register float *fa03 = fa + 2*VECTOR_WIDTH;
    register float *fa04 = fa + 3*VECTOR_WIDTH;
    register float *fa05 = fa + 4*VECTOR_WIDTH;
    register float *fa06 = fa + 5*VECTOR_WIDTH;
    register float *fa07 = fa + 6*VECTOR_WIDTH;
    register float *fa08 = fa + 7*VECTOR_WIDTH;
    //register float *fa09 = fa + 8*VECTOR_WIDTH;
    //register float *fa10 = fa + 9*VECTOR_WIDTH;
    //register float *fa11 = fa + 10*VECTOR_WIDTH;
    //register float *fa12 = fa + 11*VECTOR_WIDTH;
    //register float *fa13 = fa + 12*VECTOR_WIDTH;
    //register float *fa14 = fa + 13*VECTOR_WIDTH;
    //register float *fa15 = fa + 14*VECTOR_WIDTH;
    //register float *fa16 = fa + 15*VECTOR_WIDTH;
    //register float *fa17 = fa + 16*VECTOR_WIDTH;
    //register float *fa18 = fa + 17*VECTOR_WIDTH;
    //register float *fa19 = fa + 18*VECTOR_WIDTH;
    //register float *fa20 = fa + 19*VECTOR_WIDTH;
    //register float *fa21 = fa + 20*VECTOR_WIDTH;
    //register float *fa22 = fa + 21*VECTOR_WIDTH;
    //register float *fa23 = fa + 22*VECTOR_WIDTH;
    //register float *fa24 = fa + 23*VECTOR_WIDTH;
    
    unsigned long long i;
    int j;
#pragma nounroll
    for(i=0; i<n_trials; i++){
#pragma omp simd

        for(j=0; j<VECTOR_WIDTH; j++){
            fa01[j]=fa01[j]*fb[j] + fc[j];
            fa02[j]=fa02[j]*fb[j] + fc[j];
            fa03[j]=fa03[j]*fb[j] + fc[j];
            fa04[j]=fa04[j]*fb[j] + fc[j];
            fa05[j]=fa05[j]*fb[j] + fc[j];
            fa06[j]=fa06[j]*fb[j] + fc[j];
            fa07[j]=fa07[j]*fb[j] + fc[j];
            fa08[j]=fa08[j]*fb[j] + fc[j];
            //fa09[j]=fa09[j]*fb[j] + fc[j];
            //fa10[j]=fa10[j]*fb[j] + fc[j];
            //fa11[j]=fa11[j]*fb[j] + fc[j];
            //fa12[j]=fa12[j]*fb[j] + fc[j];
            //fa13[j]=fa13[j]*fb[j] + fc[j];
            //fa14[j]=fa14[j]*fb[j] + fc[j];
            //fa15[j]=fa15[j]*fb[j] + fc[j];
            //fa16[j]=fa16[j]*fb[j] + fc[j];
            //fa17[j]=fa17[j]*fb[j] + fc[j];
            //fa18[j]=fa18[j]*fb[j] + fc[j];
            //fa19[j]=fa19[j]*fb[j] + fc[j];
            //fa20[j]=fa20[j]*fb[j] + fc[j];
            //fa21[j]=fa21[j]*fb[j] + fc[j];
            //fa22[j]=fa22[j]*fb[j] + fc[j];
            //fa23[j]=fa23[j]*fb[j] + fc[j];
            //fa24[j]=fa24[j]*fb[j] + fc[j];

        }        
    }
    fa[0:VECTOR_WIDTH*n_chained_fmas] *= 2.0  ; // to prevent dead code elimination
} // end OMP_PARALLEL
const double t1=omp_get_wtime();
const float gflops=1.0e-9*(float)VECTOR_WIDTH*(float)n_trials*(float)flops_per_calc*(float)omp_get_max_threads()*(float)n_chained_fmas;
printf("threads:%d Chained FMAs=%d, vector width=%d, GFLOPs=%.1f, time=%.6f s, performance=%.1f GFLOP/s\n",
omp_get_max_threads(),n_chained_fmas,VECTOR_WIDTH,gflops, t1-t0, gflops/(t1-t0));


} // end main
