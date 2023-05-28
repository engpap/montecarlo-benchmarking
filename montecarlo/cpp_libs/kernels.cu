#include <stdlib.h>
#include <stdio.h>
#include <helper_cuda.h>
#include <curand_kernel.h>
#include "MonteCarlo_reduction.cuh"

typedef float real;

// GPU outputs before CPU postprocessing
typedef struct
{
    real Expected;
    real Confidence;
} __TOptionValue;

const int THREAD_N = 256;

////////////////////////////////////////////////////////////////////////////////
// Overloaded shortcut payoff functions for different precision modes
////////////////////////////////////////////////////////////////////////////////
__device__ inline float endCallValue(float S, float X, float r, float MuByT, float VBySqrtT)
{
    float callValue = S * __expf(MuByT + VBySqrtT * r) - X;
    return (callValue > 0.0F) ? callValue : 0.0F;
}

__device__ inline double endCallValue(double S, double X, double r, double MuByT, double VBySqrtT)
{
    double callValue = S * exp(MuByT + VBySqrtT * r) - X;
    return (callValue > 0.0) ? callValue     : 0.0;
}


static __global__ void rngSetupStates(
    unsigned int *d, 
    unsigned int *v, 
    int *boxmuller_flag, 
    int *boxmuller_flag_double, 
    float *boxmuller_extra, 
    double *boxmuller_extra_double, 
    int device_id)
{ 
    // determine global thread id 
    int tid = threadIdx.x + blockIdx.x * blockDim.x; 
    curandState rngState; 
    // TODO: check if this assignment is really necessary
    /*rngState.d = d[tid]; 
    for (int i = 0; i < 5; i++) rngState.v[i] = v[5*tid+i]; 
    rngState.boxmuller_flag = boxmuller_flag[tid]; 
    rngState.boxmuller_flag_double = boxmuller_flag_double[tid]; 
    rngState.boxmuller_extra = boxmuller_extra[tid]; 
    rngState.boxmuller_extra_double = boxmuller_extra_double[tid];
    */ 
    // Each threadblock gets different seed, 
    // Threads within a threadblock get different sequence numbers 
    curand_init(blockIdx.x + gridDim.x * device_id, threadIdx.x, 0, &rngState); 
    d[tid] = rngState.d; 
    for (int i = 0; i < 5; i++) v[5*tid+i] = rngState.v[i]; 
    boxmuller_flag[tid] = rngState.boxmuller_flag; 
    boxmuller_flag_double[tid] = rngState.boxmuller_flag_double; 
    boxmuller_extra[tid] = rngState.boxmuller_extra; 
    boxmuller_extra_double[tid] = rngState.boxmuller_extra_double; 
}


static __global__ void MonteCarloOneBlockPerOption( 
    unsigned int * __restrict d, 
    unsigned int * __restrict v, 
    int * __restrict boxmuller_flag, 
    int * __restrict boxmuller_flag_double,  
    float * __restrict boxmuller_extra, 
    double * __restrict boxmuller_extra_double, 
    float * optionData_S, 
    float * optionData_X, 
    float * optionData_MuByT, 
    float * optionData_VBySqrtT, 
    float * callValue_Expected, 
    float * callValue_Confidence, 
    int pathN, 
    int optionN) 
{ 
    const int SUM_N = THREAD_N; 
    __shared__ real s_SumCall[SUM_N]; 
    __shared__ real s_Sum2Call[SUM_N]; 
    // determine global thread id 
    int tid = threadIdx.x + blockIdx.x * blockDim.x; 
    // reassemble curandState 
    curandState localState; 
    localState.d = d[tid]; 
    for (int i = 0; i < 5; i++) localState.v[i] = v[5*tid+i]; 
    localState.boxmuller_flag = boxmuller_flag[tid]; 
    localState.boxmuller_flag_double = boxmuller_flag_double[tid]; 
    localState.boxmuller_extra = boxmuller_extra[tid]; 
    localState.boxmuller_extra_double = boxmuller_extra_double[tid]; 
    for (int optionIndex = blockIdx.x; optionIndex < optionN; optionIndex += gridDim.x) 
    { 
        const real S = optionData_S[optionIndex]; 
        const real X = optionData_X[optionIndex]; 
        const real MuByT = optionData_MuByT[optionIndex]; 
        const real VBySqrtT = optionData_VBySqrtT[optionIndex]; 
        for (int iSum = threadIdx.x; iSum < SUM_N; iSum += blockDim.x) 
        { 
            __TOptionValue sumCall = {0, 0}; 
#pragma unroll 8 
            for (int i = iSum; i < pathN; i += SUM_N) 
            { 
                real r = curand_normal(&localState); 
                real callValue = endCallValue(S, X, r, MuByT, VBySqrtT); 
                sumCall.Expected += callValue; 
                sumCall.Confidence += callValue * callValue; 
            } 
            s_SumCall[iSum] = sumCall.Expected; 
            s_Sum2Call[iSum] = sumCall.Confidence; 
        } 
        // Reduce shared memory accumulators 
        // and write final result to global memory 
        sumReduce<real, SUM_N, THREAD_N>(s_SumCall, s_Sum2Call); 
        if (threadIdx.x == 0) 
        { 
            __TOptionValue t = {s_SumCall[0], s_Sum2Call[0]}; 
            callValue_Expected[optionIndex] = t.Expected; 
            callValue_Confidence[optionIndex] = t.Confidence; 
        } 
    } 
}

