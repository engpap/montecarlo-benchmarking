package com.polimi.montecarlo;

import org.graalvm.polyglot.Value;

public class MonteCarlo extends Benchmark{

    private static final String RNG_SETUP_STATES_KERNEL = ""+
    "static __global__ void rngSetupStates("+
    "    curandState *rngState,"+
    "    int device_id)"+
    "{"+
    "    // determine global thread id"+
    "    int tid = threadIdx.x + blockIdx.x * blockDim.x;"+
    "    // Each threadblock gets different seed,"+
    "    // Threads within a threadblock get different sequence numbers"+
    "    curand_init(blockIdx.x + gridDim.x * device_id, threadIdx.x, 0, &rngState[tid]);"+
    "}";

    private static final String MONTECARLO_ONE_BLOCK_PER_OPTION_KERNEL = ""+
    "static __global__ void MonteCarloOneBlockPerOption("+
    "    curandState * __restrict rngStates,"+
    "    const __TOptionData * __restrict d_OptionData,"+
    "    __TOptionValue * __restrict d_CallValue,"+
    "    int pathN,"+
    "    int optionN)"+
    "{"+
    "    const int SUM_N = THREAD_N;"+
    "    __shared__ real s_SumCall[SUM_N];"+
    "    __shared__ real s_Sum2Call[SUM_N];"+
    ""+
    "    // determine global thread id"+
    "    int tid = threadIdx.x + blockIdx.x * blockDim.x;"+
    ""+
    "    // Copy random number state to local memory for efficiency"+
    "    curandState localState = rngStates[tid];"+
    "    for(int optionIndex = blockIdx.x; optionIndex < optionN; optionIndex += gridDim.x)"+
    "    {"+
    "        const real        S = d_OptionData[optionIndex].S;"+
    "        const real        X = d_OptionData[optionIndex].X;"+
    "        const real    MuByT = d_OptionData[optionIndex].MuByT;"+
    "        const real VBySqrtT = d_OptionData[optionIndex].VBySqrtT;"+
    "        "+
    "        //Cycle through the entire samples array:"+
    "        //derive end stock price for each path"+
    "        //accumulate partial integrals into intermediate shared memory buffer"+
    "        for (int iSum = threadIdx.x; iSum < SUM_N; iSum += blockDim.x)"+
    "        {"+
    "            __TOptionValue sumCall = {0, 0};"+
    "            "+
    "            #pragma unroll 8"+
     "           for (int i = iSum; i < pathN; i += SUM_N)"+
    "            {"+
    "                real              r = curand_normal(&localState);"+
    "                real      callValue = endCallValue(S, X, r, MuByT, VBySqrtT);"+
    "                sumCall.Expected   += callValue;"+
    "                sumCall.Confidence += callValue * callValue;"+
    "            }"+
    "            "+
     "           s_SumCall[iSum]  = sumCall.Expected;"+
     "           s_Sum2Call[iSum] = sumCall.Confidence;"+
    "        }"+
    "         "+
    "        //Reduce shared memory accumulators"+
    "        //and write final result to global memory"+
    "        sumReduce<real, SUM_N, THREAD_N>(s_SumCall, s_Sum2Call);"+
    "        "+
    "       if (threadIdx.x == 0)"+
    "        {"+
    "          __TOptionValue t = {s_SumCall[0], s_Sum2Call[0]};"+
    "            d_CallValue[optionIndex] = t;"+
    "       }"+
    "    }"+
    "}";

    /// Kernels
    private Value monteCarloOneBlockPerOptionKernelFunction;
    private Value rngSetupStatesKernelFunction;

    /// __TOptionData
    private Value optionData_S, optionData_X, optionData_MuByT, optionData_VBySqrtT;

    /// __TOptionValue
    private Value optionValue_Expected, optionValue_Confidence;

    /// rngStates
    private Value rngStates;

    /// Constants
    private static final int THREAD_N = 256;

    public MonteCarlo(BenchmarkConfig currentConfig) {super(currentConfig);}

    @Override
    protected void initializeTest(int iteration) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'initializeTest'");
    }

    /**
     * Resembles initMonteCarloGPU.
     * 
    */ 
    @Override
    protected void allocateTest(int iteration) {

        // Allocate __TOptionData's arrays: equivalent to allocate plan->um_OptionData,
        optionData_S = requestArray("float", config.size);
        optionData_X = requestArray("float", config.size);
        optionData_MuByT = requestArray("float", config.size);
        optionData_VBySqrtT = requestArray("float", config.size);

        // Allocate __TOptionValue's arrays: equivalent to allocate plan->um_CallValue,
        optionValue_Expected = requestArray("float", config.size);
        optionValue_Confidence = requestArray("float", config.size);

        /// TODO: Allocate rngStates array
        //rngStates = requestArray("", config.numBlocks*THREAD_N);

        // Context initialization
        Value buildKernel = context.eval("grcuda", "buildkernel");
        
        // TODO: kernel modification and then exec call
        //rngSetupStatesKernelFunction = buildKernel.execute(RNG_SETUP_STATES_KERNEL, "rngSetupStates", "pointer, sint32");

    }












    @Override
    protected void resetIteration(int iteration) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'resetIteration'");
    }

    @Override
    protected void runTest(int iteration) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'runTest'");
    }

    @Override
    protected void cpuValidation() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'cpuValidation'");
    }
    
}
