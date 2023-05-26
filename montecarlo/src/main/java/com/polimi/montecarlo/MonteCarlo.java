package com.polimi.montecarlo;

import static org.junit.Assert.assertTrue;

import java.io.File;

import org.graalvm.polyglot.Value;

public class MonteCarlo extends Benchmark{

    /// Kernels
    private Value monteCarloOneBlockPerOptionKernelFunction;
    private Value rngSetupStatesKernelFunction;

    /// Constants
    private static final int THREAD_N = 256;
    private static final int Max_OPTIONS = 1024*1024*64;

    private int OPT_N;

    /// OptionData and CallValueGPU are arrays of OPT_N elements
    private OptionData[] optionData;
    private OptionValue[] callValueGPU;

    private Plan[] plan;

    /// rngStates is an array of curandState states, thus we decompose curandState struct into its components
    /// curandStateXORWOW
    private Value curandState_d, curandState_v, curandState_boxmuller_flag, curandState_boxmuller_flag_double, curandState_boxmuller_extra, curandState_boxmuller_extra_double;

    public MonteCarlo(BenchmarkConfig currentConfig) {
        super(currentConfig);
        init();
    }

    /**
     * This method resembles implementation of intialization functions of MonteCarloMultiGPU.cpp
     */
    private void init() { // TRANSLATION FINISHED!

        boolean strongScaling;
        if(config.scalingChoice.isEmpty()){
            strongScaling=false;
        }
        else{
            if(config.scalingChoice.equalsIgnoreCase("strong"))
                strongScaling=true;
            else
                strongScaling=false;
        }
        int nOptions = config.size * config.size;
        // adjustProblemSize not implemented because size can be adjusted by config file
        int scale = (strongScaling) ? 1 : config.numGpus;
        OPT_N = nOptions * scale;
        int PATH_N = 262144;

        optionData = new OptionData[OPT_N];
        callValueGPU = new OptionValue[OPT_N];
       
        float S, X, T, R, V;
        float Expected, Confidence;
        for (int i = 0; i < OPT_N; i++) {
            S = Utils.randFloat(5.0f, 50.0f);
            X = Utils.randFloat(10.0f, 25.0f);
            T = Utils.randFloat(1.0f, 5.0f);
            R = 0.06f;
            V = 0.10f;
            optionData[i] = new OptionData(S, X, T, R, V);

            Expected = -1.0f;
            Confidence = -1.0f;
            callValueGPU[i] = new OptionValue(Expected, Confidence);
        }

        // Instantiate the plan array corresponding to TOptionPlan[GPU_N] (optionSolver)
        this.plan = new Plan[config.numGpus];
        
        // Get option count for each GPU
        for(int gpu_index = 0; gpu_index < config.numGpus; gpu_index++)
        {
            plan[gpu_index].setOptionCount(OPT_N / config.numGpus);
        }

        // Take into account cases with "odd" option counts
        for(int gpu_index = 0; gpu_index < (OPT_N % config.numGpus); gpu_index++)
        {
            plan[gpu_index].setOptionCount(plan[gpu_index].getOptionCount() + 1);
        }

        //Assign GPU option ranges
        int gpuBase = 0;
        for(int gpu_index=0; gpu_index < config.numGpus; gpu_index++)
        {
            plan[gpu_index].setDevice(gpu_index);
            plan[gpu_index].setOptionData(optionData, gpuBase, gpuBase + plan[gpu_index].getOptionCount()); //TODO: WRONG
            plan[gpu_index].setCallValue(callValueGPU, gpuBase, gpuBase + plan[gpu_index].getOptionCount());
            plan[gpu_index].setPathN(PATH_N);
            plan[gpu_index].setGridSize(adjustGridSize(plan[gpu_index].getDevice(), plan[gpu_index].getOptionCount()));
            gpuBase += plan[gpu_index].getOptionCount();
        }
    }

    //TODO adjustGridSize
    private int adjustGridSize(int device, int optionCount) {
        // TODO: implement adjustGridSize when binding is implemented
        return 1; //dummy return, to remove
    }

    /// Resembles initMonteCarloGPU.
    @Override
    protected void allocateTest(int iteration) {

        // Allocate optionData and optionValue (callValue)
        for(int gpu_index=0; gpu_index < config.numGpus; gpu_index++){
            plan[gpu_index].setS(requestArray("float", config.size));
            plan[gpu_index].setX(requestArray("float", config.size));
            plan[gpu_index].setMuByT(requestArray("float", config.size));
            plan[gpu_index].setVBySqrtT(requestArray("float", config.size));
            plan[gpu_index].setExpected(requestArray("float", config.size));
            plan[gpu_index].setConfidence(requestArray("float", config.size));
        }
       

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

        // TODO: remove this
        File f = new File("/home/ubuntu/montecarlo-benchmarking/montecarlo/src/main/java/com/polimi/montecarlo/kernels.ptx");
        if(f.exists() && !f.isDirectory()) { 
            System.out.println("File exists!");
        } else {
            System.out.println("File doesn't exist");
        }
        String path = "/home/ubuntu/montecarlo-benchmarking/montecarlo/src/main/java/com/polimi/montecarlo/kernels.ptx";

        Value cu = context.eval("grcuda", "CU");

        rngSetupStatesKernelFunction = cu.invokeMember("bindkernel", path, "cxx rngSetupStates(" +
        "d: out pointer uint32, " + 
        "v: out pointer uint32, " + 
        "boxmuller_flag: out pointer sint32, " + 
        "boxmuller_flag_double: out pointer sint32, " + 
        "boxmuller_extra: out pointer float, " + 
        "boxmuller_extra_double: out pointer double, " + 
        "device_id: sint32)");

        monteCarloOneBlockPerOptionKernelFunction = cu.invokeMember("bindkernel", path, "cxx MonteCarloOneBlockPerOption( " +
        "d: int pointer uint32, " + 
        "v in pointer uint32, " + 
        "boxmuller_flag: in pointer sint32, " + 
        "boxmuller_flag_double: in pointer sint32, " +
        "boxmuller_extra: in pointer float, " + 
        "boxmuller_extra_double: in pointer double, " + 
        "optionData_S: in pointer float, " + 
        "optionData_X: in pointer float, " +  
        "optionData_VBySqrtT: in pointer float, " +  
        "optionData_callValue_Expected: out pointer float, " +  
        "optionData_callValue_Confidence: out pointer float, " +
        "pathN: sint32" +
        "optionN: sint32)"); 

        
    }

    @Override
    protected void initializeTest(int iteration) {
    if(config.debug)
        System.out.println("    INSIDE initializeTest() - " + iteration);
        rngSetupStatesKernelFunction.execute(config.numBlocks, config.blockSize1D)
                .execute(curandState_d, curandState_v, curandState_boxmuller_flag, curandState_boxmuller_flag_double, curandState_boxmuller_extra, curandState_boxmuller_extra_double, config.deviceId);
    }

    @Override
    protected void resetIteration(int iteration) {
        System.out.println("test");
        // TODO Auto-generated method stub
        //throw new UnsupportedOperationException("Unimplemented method 'resetIteration'");
    }

    @Override
    protected void runTest(int iteration) {
        int optionStart=0;
        // For each GPU, we initialize the option data values for the current iteration
        for(int gpu_index=0; gpu_index < config.numGpus; gpu_index ++){

            if(plan[gpu_index].getOptionCount() <= 0 || plan[gpu_index].getOptionCount() > Max_OPTIONS){
            System.out.println("Invalid optN value");
            return;
            }

            double MuByT, VBySqrtT;
            // This loop iterates over the options assigned to the current GPU
            for(int option_index=optionStart; option_index < optionStart + plan[gpu_index].getOptionCount(); option_index++){
                MuByT = (optionData[option_index].getR() - (0.5 * optionData[option_index].getV() * optionData[gpu_index].getV())) * optionData[gpu_index].getT();
                VBySqrtT = optionData[option_index].getV() * Math.sqrt(optionData[gpu_index].getT());
                
                plan[gpu_index].getS().setArrayElement(option_index, optionData[option_index].getS());
                plan[gpu_index].getX().setArrayElement(option_index, optionData[option_index].getX());
                plan[gpu_index].getMuByT().setArrayElement(option_index, MuByT);
                plan[gpu_index].getVBySqrtT().setArrayElement(option_index, VBySqrtT);
            }
            optionStart += plan[gpu_index].getOptionCount();

            // TODO: revise kernel to take into account the offset
            monteCarloOneBlockPerOptionKernelFunction.execute(config.numBlocks, config.blockSize1D)
            .execute(optionData[gpu_index], curandState_v, curandState_boxmuller_flag, curandState_boxmuller_flag_double, curandState_boxmuller_extra, curandState_boxmuller_extra_double, config.deviceId);
        }



    }


    /**
     * This method execute Black-Scholes computation on the CPU and compares the MonteCarlo results generated with the GPUs.
     */
    @Override
    protected void cpuValidation() {
        System.out.println(">>> Comparing Monte Carlo and Black-Scholes results...");
        float[] callValueBS = new float[OPT_N];
        float delta, ref;

        float sumDelta = 0;
        float sumRef = 0;
        float sumReserve = 0;

        // Iterate through all plans and compare BlackScholes results for each option
        for(int gpu_index = 0; gpu_index < config.numGpus; gpu_index++)
        {
            for (int option_index = 0; option_index < plan[gpu_index].getOptionCount(); option_index ++)
            {
                callValueBS[option_index] = Utils.BlackScholesCall(plan[gpu_index].getOptionData()[option_index]);
                delta = Math.abs(callValueBS[option_index] - plan[gpu_index].getCallValue()[option_index].getExpected());
                ref = callValueBS[option_index];
                sumDelta += delta;
                sumRef = Math.abs(ref);                
                if (delta > 1e-6f) {
                    sumReserve += plan[gpu_index].getCallValue()[option_index].getConfidence() / delta;
                }
            }           
        }    
        sumReserve /= OPT_N;

        System.out.println("Test Summary...");
        System.out.println("L1 norm        : "+ (sumDelta / sumRef));
        System.out.println("Average reserve: "+ sumReserve);
        assertTrue("Test passed", sumReserve > 1.0f);
    }



    
}