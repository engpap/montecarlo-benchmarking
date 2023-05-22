package com.polimi.montecarlo;

import org.graalvm.polyglot.Value;

public class MonteCarlo extends Benchmark{

    private static final String RNG_SETUP_STATES_KERNEL = "" +
"extern \"C\" __global__ void rngSetupStates(\n" +
"    unsigned int *d,\n" +
"    unsigned int *v,\n" +
"    int *boxmuller_flag,\n" +
"    int *boxmuller_flag_double,\n" +
"    float *boxmuller_extra,\n" +
"    double *boxmuller_extra_double,\n" +
"    int device_id)\n" +
"{\n" +
"    // determine global thread id\n" +
"    int tid = threadIdx.x + blockIdx.x * blockDim.x;\n" +
"    curandState rngState;\n" +
"    rngState.d = d[tid];\n" +
"    for (int i = 0; i < 5; i++) rngState.v[i] = v[5*tid+i];\n" +
"    rngState.boxmuller_flag = boxmuller_flag[tid];\n" +
"    rngState.boxmuller_flag_double = boxmuller_flag_double[tid];\n" +
"    rngState.boxmuller_extra = boxmuller_extra[tid];\n" +
"    rngState.boxmuller_extra_double = boxmuller_extra_double[tid];\n" +
"    // Each threadblock gets different seed,\n" +
"    // Threads within a threadblock get different sequence numbers\n" +
"    curand_init(blockIdx.x + gridDim.x * device_id, threadIdx.x, 0, &rngState);\n" +
"    d[tid] = rngState.d;\n" +
"    for (int i = 0; i < 5; i++) v[5*tid+i] = rngState.v[i];\n" +
"    boxmuller_flag[tid] = rngState.boxmuller_flag;\n" +
"    boxmuller_flag_double[tid] = rngState.boxmuller_flag_double;\n" +
"    boxmuller_extra[tid] = rngState.boxmuller_extra;\n" +
"    boxmuller_extra_double[tid] = rngState.boxmuller_extra_double;\n" +
"}";


    private static final String MONTECARLO_ONE_BLOCK_PER_OPTION_KERNEL =
    " static __global__ void MonteCarloOneBlockPerOption(\n" +
    "    unsigned int * __restrict d,\n" +
    "    unsigned int * __restrict v,\n" +
    "    int * __restrict boxmuller_flag,\n" +
    "    int * __restrict boxmuller_flag_double, \n" +
    "    float * __restrict boxmuller_extra,\n" +
    "    double * __restrict boxmuller_extra_double,\n" +
    "    real * __restrict optionData_S,\n" +
    "    real * __restrict optionData_X,\n" +
    "    real * __restrict optionData_MuByT,\n" +
    "    real * __restrict optionData_VBySqrtT,\n" +
    "    real * __restrict callValue_Expected,\n" +
    "    real * __restrict callValue_Confidence,\n" +
    "    int pathN,\n" +
    "    int optionN)\n" +
    "{\n" +
    "    const int SUM_N = THREAD_N;\n" +
    "    __shared__ real s_SumCall[SUM_N];\n" +
    "    __shared__ real s_Sum2Call[SUM_N];\n" +
    "    // determine global thread id\n" +
    "    int tid = threadIdx.x + blockIdx.x * blockDim.x;\n" +
    "    // reassemble curandState\n" +
    "    curandState localState;\n" +
    "    localState.d = d[tid];\n" +
    "    for (int i = 0; i < 5; i++) localState.v[i] = v[5*tid+i];\n" +
    "    localState.boxmuller_flag = boxmuller_flag[tid];\n" +
    "    localState.boxmuller_flag_double = boxmuller_flag_double[tid];\n" +
    "    localState.boxmuller_extra = boxmuller_extra[tid];\n" +
    "    localState.boxmuller_extra_double = boxmuller_extra_double[tid];\n" +
    "    for (int optionIndex = blockIdx.x; optionIndex < optionN; optionIndex += gridDim.x)\n" +
    "    {\n" +
    "        const real S = optionData_S[optionIndex];\n" +
    "        const real X = optionData_X[optionIndex];\n" +
    "        const real MuByT = optionData_MuByT[optionIndex];\n" +
    "        const real VBySqrtT = optionData_VBySqrtT[optionIndex];\n" +
    "        for (int iSum = threadIdx.x; iSum < SUM_N; iSum += blockDim.x)\n" +
    "        {\n" +
    "            __TOptionValue sumCall = {0, 0};\n" +
    "#pragma unroll 8\n" +
    "            for (int i = iSum; i < pathN; i += SUM_N)\n" +
    "            {\n" +
    "                real r = curand_normal(&localState);\n" +
    "                real callValue = endCallValue(S, X, r, MuByT, VBySqrtT);\n" +
    "                sumCall.Expected += callValue;\n" +
    "                sumCall.Confidence += callValue * callValue;\n" +
    "            }\n" +
    "            s_SumCall[iSum] = sumCall.Expected;\n" +
    "            s_Sum2Call[iSum] = sumCall.Confidence;\n" +
    "        }\n" +
    "        // Reduce shared memory accumulators\n" +
    "        // and write final result to global memory\n" +
    "        sumReduce<real, SUM_N, THREAD_N>(s_SumCall, s_Sum2Call);\n" +
    "        if (threadIdx.x == 0)\n" +
    "        {\n" +
    "            __TOptionValue t = {s_SumCall[0], s_Sum2Call[0]};\n" +
    "            callValue_Expected[optionIndex] = t.Expected;\n" +
    "            callValue_Confidence[optionIndex] = t.Confidence;\n" +
    "        }\n" +
    "    }\n" +
    "}";


    /// Kernels
    private Value monteCarloOneBlockPerOptionKernelFunction;
    private Value rngSetupStatesKernelFunction;

    /// Constants
    private static final int THREAD_N = 256;

    /// __TOptionData
    private Value optionData_S, optionData_X, optionData_MuByT, optionData_VBySqrtT;

    /// __TOptionValue
    private Value optionValue_Expected, optionValue_Confidence;

    /// rngStates: array of curandState states, thus we decompose curandState struct into its components
    /// curandStateXORWOW
    private Value curandState_d, curandState_v, curandState_boxmuller_flag, curandState_boxmuller_flag_double, curandState_boxmuller_extra, curandState_boxmuller_extra_double;

    public MonteCarlo(BenchmarkConfig currentConfig) {super(currentConfig);}

    @Override
    protected void initializeTest(int iteration) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'initializeTest'");
    }

    /// Resembles initMonteCarloGPU.
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

        /* GrCUDA does not directly support unsigned integer data types, as Java itself doesn't have built-in support for unsigned integers.
        However, WE can work around this by using a larger data type. For example, we can use long to represent an unsigned int from C++.
        This allows you to use the full range of unsigned int values from C++, since a long in Java has more bits than an int and thus can 
        represent larger values. */
        curandState_d = requestArray("long", config.numBlocks * THREAD_N);
        curandState_v = requestArray("long", config.numBlocks * THREAD_N * 5); // v[5] is an array of 5 elements, hence config.size * 5
        curandState_boxmuller_flag = requestArray("int", config.numBlocks * THREAD_N);
        curandState_boxmuller_flag_double = requestArray("int", config.numBlocks * THREAD_N);
        curandState_boxmuller_extra = requestArray("float", config.numBlocks * THREAD_N);
        curandState_boxmuller_extra_double = requestArray("double", config.numBlocks * THREAD_N);

        // Context initialization
        Value buildKernel = context.eval("grcuda", "buildkernel");
        
        // Build RNG_SETUP_STATES_KERNEL
        rngSetupStatesKernelFunction = buildKernel.execute(RNG_SETUP_STATES_KERNEL, "rngSetupStates", "pointer, pointer, pointer, pointer, pointer, pointer, sint32");
        monteCarloOneBlockPerOptionKernelFunction = buildKernel.execute(MONTECARLO_ONE_BLOCK_PER_OPTION_KERNEL, "monteCarloOneBlockPerOption", "pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer, sint32, sint32");

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