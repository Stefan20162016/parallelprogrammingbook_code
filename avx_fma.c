// icpc -qopenmp -xHost -S  avx_fma.c && time ./avx_fma
// -qopt-report=5 -qopt-report-phase=vec -qopt-report-embed
// export KMP_HW_SUBSET=1t # to disable hyperthreading
// export KMP_AFFINITY=compact,1 # core affinity
#include <stdio.h>
#include <omp.h>

#define VECTOR_WIDTH 4
const int n_trials=2*1000000000;
const int flops_per_calc =2;
const int n_chained_fmas=10;

int main(){
#pragma omp parallel
{ } // warm-up threads
//omp_set_num_threads(8);
const double t0=omp_get_wtime();

#pragma omp parallel
{ 
    char x=1,x2=1;
    double fa[VECTOR_WIDTH*n_chained_fmas];
    double fb[VECTOR_WIDTH];
    double fc[VECTOR_WIDTH];
    fa[0:VECTOR_WIDTH*n_chained_fmas]=0.0;
    fb[0:VECTOR_WIDTH]=0.5;
    fc[0:VECTOR_WIDTH]=1.0;
    char y=x*2;
    register double *fa01 = fa + 0*VECTOR_WIDTH;
    register double *fa02 = fa + 1*VECTOR_WIDTH;
    register double *fa03 = fa + 2*VECTOR_WIDTH;
    register double *fa04 = fa + 3*VECTOR_WIDTH;
    register double *fa05 = fa + 4*VECTOR_WIDTH;
    register double *fa06 = fa + 5*VECTOR_WIDTH;
    register double *fa07 = fa + 6*VECTOR_WIDTH;
    register double *fa08 = fa + 7*VECTOR_WIDTH;
    register double *fa09 = fa + 8*VECTOR_WIDTH;
    register double *fa10 = fa + 9*VECTOR_WIDTH;
    //register double *fa11 = fa + 10*VECTOR_WIDTH;
    //register double *fa12 = fa + 11*VECTOR_WIDTH;
    //register double *fa13 = fa + 12*VECTOR_WIDTH;
    //register double *fa14 = fa + 13*VECTOR_WIDTH;

    int i,j;
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
            fa09[j]=fa09[j]*fb[j] + fc[j];
            fa10[j]=fa10[j]*fb[j] + fc[j];
            //fa11[j]=fa11[j]*fb[j] + fc[j];
            //fa12[j]=fa12[j]*fb[j] + fc[j];
            //fa13[j]=fa13[j]*fb[j] + fc[j];
            //fa14[j]=fa14[j]*fb[j] + fc[j];
        }        
    }
    fa[0:VECTOR_WIDTH*n_chained_fmas] *= 2.0; // to prevent dead code elimination
} // end OMP_PARALLEL
const double t1=omp_get_wtime();
const double gflops=1.0e-9*(double)VECTOR_WIDTH*(double)n_trials*(double)flops_per_calc*(double)omp_get_max_threads()*(double)n_chained_fmas;
printf("threads:%d Chained FMAs=%d, vector width=%d, GFLOPs=%.1f, time=%.6f s, performance=%.1f GFLOP/s\n",
omp_get_max_threads(),n_chained_fmas,VECTOR_WIDTH,gflops, t1-t0, gflops/(t1-t0));


} // end main