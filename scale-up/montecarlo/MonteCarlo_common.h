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

#ifndef MONTECARLO_COMMON_H
#define MONTECARLO_COMMON_H
#include "realtype.h"
#include "curand_kernel.h"

////////////////////////////////////////////////////////////////////////////////
// Global types
////////////////////////////////////////////////////////////////////////////////

// Holds data for a financial option.
typedef struct
{
    float S; /// stock price
    float X; /// strike price
    float T; /// time to expiration 
    float R; /// risk-free interest rate
    float V; /// volatility 
} TOptionData;

/// Holds the expected value and confidence level of an option.
typedef struct
        //#ifdef __CUDACC__
        //__align__(8)
        //#endif
{
    float Expected;
    float Confidence;
} TOptionValue;

// GPU outputs before CPU postprocessing;
/// Similar to TOptionValue, but with a different precision type.
typedef struct
{
    real Expected;
    real Confidence;
} __TOptionValue;



/// Holds information about a Monte Carlo simulation plan.
/// "device-side" refers to data that is stored and processed on the GPU,
/// while "host-side" refers to data that is stored and processed on the CPU.
typedef struct
{
    //Device ID for multi-GPU version
    int device;
    //Option count for this plan
    int optionCount;

    //Host-side data source and result destination
    TOptionData  *optionData;
    TOptionValue *callValue;

    //Temporary Host-side pinned memory for async + faster data transfers
    __TOptionValue *h_CallValue;

    // Device- and host-side option data
    void * d_OptionData;
    void * h_OptionData;

    // Device-side option values
    void * d_CallValue;

    //Intermediate device-side buffers
    void *d_Buffer;    

    //random number generator states
    /*
    rngStates is an array of curandState structures. 
    These structures hold the random number generator (RNG) states used in the Monte Carlo algorithm.
    RNGs are essential in Monte Carlo simulations because they generate random samples to estimate 
    the expected value of a function.
    */
    curandState *rngStates;

    //Pseudorandom samples count
    int pathN;

    //Time stamp
    float time;

    int gridSize;
} TOptionPlan;


extern "C" void initMonteCarloGPU(TOptionPlan *plan);

// It performs the Monte Carlo simulation on the GPU.
// A stream is a sequence of operations that are performed in order on the device.
// Streams allows independent concurrent in-order queues of execution.
extern "C" void MonteCarloGPU(TOptionPlan *plan, cudaStream_t stream=0);

extern "C" void closeMonteCarloGPU(TOptionPlan *plan);

#endif
