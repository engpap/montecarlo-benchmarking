/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////
// Global types
////////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <helper_cuda.h>
#include <curand_kernel.h>
#include "MonteCarlo_common.h"

////////////////////////////////////////////////////////////////////////////////
// Helper reduction template
// Please see the "reduction" CUDA Sample for more information
////////////////////////////////////////////////////////////////////////////////
#include "MonteCarlo_reduction.cuh"

////////////////////////////////////////////////////////////////////////////////
// Internal GPU-side data structures
////////////////////////////////////////////////////////////////////////////////
#define MAX_OPTIONS (1024 * 1024*32)

// Preprocessed input option data
typedef struct
{
    real S;
    real X;
    real MuByT;
    real VBySqrtT;
} __TOptionData;

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
    return (callValue > 0.0) ? callValue : 0.0;
}

#define THREAD_N 256

////////////////////////////////////////////////////////////////////////////////
// This kernel computes the integral over all paths using a single thread block
// per option. It is fastest when the number of thread blocks times the work per
// block is high enough to keep the GPU busy.
////////////////////////////////////////////////////////////////////////////////
static __global__ void MonteCarloOneBlockPerOption(
    unsigned int *d,
    unsigned int *v,
    int *boxmuller_flag,
    int *boxmuller_flag_double, 
    float *boxmuller_extra,
    double *boxmuller_extra_double,
    real *optionData_S,
    real *optionData_X,
    real *optionData_MuByT,
    real *optionData_VBySqrtT,
    real *callValue_Expected,
    real *callValue_Confidence,
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
    rngState.d = d[tid];
    for(int i = 0; i < 5; i++)
        rngState.v[i] = v[tid * 5 + i];
    rngState.boxmuller_flag = boxmuller_flag[tid];
    rngState.boxmuller_flag_double = boxmuller_flag_double[tid];
    rngState.boxmuller_extra = boxmuller_extra[tid];
    rngState.boxmuller_extra_double = boxmuller_extra_double[tid];

    // Each threadblock gets different seed,
    // Threads within a threadblock get different sequence numbers
    curand_init(blockIdx.x + gridDim.x * device_id, threadIdx.x, 0, &rngState);
}


////////////////////////////////////////////////////////////////////////////////
// Host-side interface to GPU Monte Carlo
////////////////////////////////////////////////////////////////////////////////

extern "C" void initMonteCarloGPU(TOptionPlan *plan)
{
    checkCudaErrors(cudaMallocManaged(&plan->um_OptionData, sizeof(__TOptionData) * (plan->optionCount)));
    checkCudaErrors(cudaMallocManaged(&plan->um_CallValue, sizeof(__TOptionValue) * (plan->optionCount)));
    // Allocate internal device memory
    // Allocate states for pseudo random number generators
    checkCudaErrors(cudaMallocManaged((void **)&plan->rngStates,
                                      plan->gridSize * THREAD_N * sizeof(curandState)));

    // place each device pathN random numbers apart on the random number sequence
    rngSetupStates<<<plan->gridSize, THREAD_N>>>(plan->rngStates, plan->device);
    getLastCudaError("rngSetupStates kernel failed.\n");
}

// Compute statistics and deallocate internal device memory
extern "C" void closeMonteCarloGPU(TOptionPlan *plan)
{
    for (int i = 0; i < plan->optionCount; i++)
    {
        const double RT = plan->optionData[i].R * plan->optionData[i].T;
        const double sum = plan->um_CallValue[i].Expected;
        const double sum2 = plan->um_CallValue[i].Confidence;
        const double pathN = plan->pathN;
        // Derive average from the total sum and discount by riskfree rate
        plan->callValue[i].Expected = (float)(exp(-RT) * sum / pathN);
        // Standard deviation
        double stdDev = sqrt((pathN * sum2 - sum * sum) / (pathN * (pathN - 1)));
        // Confidence width; in 95% of all cases theoretical value lies within these borders
        plan->callValue[i].Confidence = (float)(exp(-RT) * 1.96 * stdDev / sqrt(pathN));
    }

    checkCudaErrors(cudaFree(plan->rngStates));
    checkCudaErrors(cudaFree(plan->um_CallValue));
    checkCudaErrors(cudaFree(plan->um_OptionData));
}

// Main computations
extern "C" void MonteCarloGPU(TOptionPlan *plan, cudaStream_t stream)
{

    if (plan->optionCount <= 0 || plan->optionCount > MAX_OPTIONS)
    {
        printf("MonteCarloGPU(): bad option count.\n");
        return;
    }

    __TOptionData *optionData = (__TOptionData *)plan->um_OptionData;

    for (int i = 0; i < plan->optionCount; i++)
    {
        const double T = plan->optionData[i].T;
        const double R = plan->optionData[i].R;
        const double V = plan->optionData[i].V;
        const double MuByT = (R - 0.5 * V * V) * T;
        const double VBySqrtT = V * sqrt(T);
        optionData[i].S = (real)plan->optionData[i].S;
        optionData[i].X = (real)plan->optionData[i].X;
        optionData[i].MuByT = (real)MuByT;
        optionData[i].VBySqrtT = (real)VBySqrtT;
    }

    MonteCarloOneBlockPerOption<<<plan->gridSize, THREAD_N, 0, stream>>>(
        plan->rngStates,
        (__TOptionData *)(plan->um_OptionData),
        (__TOptionValue *)(plan->um_CallValue),
        plan->pathN,
        plan->optionCount);
    getLastCudaError("MonteCarloOneBlockPerOption() execution failed\n");
}
