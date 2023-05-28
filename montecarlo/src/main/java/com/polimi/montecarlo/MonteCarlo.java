package com.polimi.montecarlo;

import static org.junit.Assert.assertTrue;

import java.io.File;

import org.graalvm.polyglot.Value;

public class MonteCarlo extends Benchmark {

    private static final String PATH_TO_BINDED_FUNCTIONS = "/home/ubuntu/montecarlo-benchmarking/montecarlo/cpp_libs/";

    /// Kernels
    private Value monteCarloOneBlockPerOptionKernelFunction;
    private Value rngSetupStatesKernelFunction;

    /// Host Functions
    private Value adjustProblemSizeFunction;
    private Value adjustGridSizeFunction;

    /// Constants
    private static final int THREAD_N = 256;
    private static final int Max_OPTIONS = 1024 * 1024 * 64;

    private int OPT_N;

    /// OptionData and CallValueGPU are arrays of OPT_N elements
    private OptionData[] optionData;
    private OptionValue[] callValueGPU;

    private Plan[] plan;

    public MonteCarlo(BenchmarkConfig currentConfig) {
        super(currentConfig);
        init();
    }

    /**
     * This method resembles implementation of intialization functions of
     * MonteCarloMultiGPU.cpp
     */
    private void init() { // TRANSLATION FINISHED!

        boolean strongScaling;
        if (config.scalingChoice.isEmpty()) {
            strongScaling = false;
        } else {
            if (config.scalingChoice.equalsIgnoreCase("strong"))
                strongScaling = true;
            else
                strongScaling = false;
        }
        int nOptions = config.size * config.size;
        setupAdjustSizeHostFunctions();
        nOptions = adjustProblemSizeFunction.execute(config.numGpus, nOptions).asInt();
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

        // Get option count for each GPU and instantiate the Plan class
        for (int gpu_index = 0; gpu_index < config.numGpus; gpu_index++) {
            if(gpu_index < (OPT_N % config.numGpus)) {
                plan[gpu_index] = new Plan((int) (OPT_N / config.numGpus) + 1);
            } else {
                plan[gpu_index] = new Plan((int) (OPT_N / config.numGpus));
            }
        }

        //TODO: remove this or remove the if above, better this since the array are instanciated in the constructor
        // Take into account cases with "odd" option counts
        //for (int gpu_index = 0; gpu_index < (OPT_N % config.numGpus); gpu_index++) {
        //    plan[gpu_index].setOptionCount(plan[gpu_index].getOptionCount() + 1);
        //}

        // Assign GPU option ranges
        int gpuBase = 0;
        for (int gpu_index = 0; gpu_index < config.numGpus; gpu_index++) {
            plan[gpu_index].setDevice(gpu_index);
            plan[gpu_index].setOptionData(optionData, gpuBase, gpuBase + plan[gpu_index].getOptionCount()); // TODO:
                                                                                                            // WRONG
            plan[gpu_index].setCallValue(callValueGPU, gpuBase, gpuBase + plan[gpu_index].getOptionCount());
            plan[gpu_index].setPathN(PATH_N);
            plan[gpu_index].setGridSize(adjustGridSizeFunction.execute(plan[gpu_index].getDevice(), plan[gpu_index].getOptionCount()).asInt());
            gpuBase += plan[gpu_index].getOptionCount();
        }
    }

    private void setupAdjustSizeHostFunctions() {
        String path_to_binded_host_functions = PATH_TO_BINDED_FUNCTIONS + "libutils.so";
        Value cu = context.eval("grcuda", "CU");
        checkFileExists(path_to_binded_host_functions);

        adjustProblemSizeFunction = cu.invokeMember("bind", path_to_binded_host_functions,
            "cxx adjustProblemSize( " +
                "GPU_N: sint32, " +
                "default_nOptions: sint32): sint32"
            );

        adjustGridSizeFunction = cu.invokeMember("bind", path_to_binded_host_functions,
            "cxx adjustGridSize( " +
                "GPUIndex: sint32, " +
                "defaultGridSize: sint32) : sint32"
            );
    }

    /**
     * This method resembles the first part of initMonteCarloGPU.
     * It allocates memory for all the arrays used in the benchmark.
     */
    @Override
    protected void allocateTest(int iteration) { // TRANSLATION FINISHED!

        // Allocate memory for optionData and optionValue (callValue)
        // Allocate memory for the curandStates
        for (int gpu_index = 0; gpu_index < config.numGpus; gpu_index++) {
            plan[gpu_index].setS(requestArray("float", plan[gpu_index].getOptionCount()));
            plan[gpu_index].setX(requestArray("float", plan[gpu_index].getOptionCount()));
            plan[gpu_index].setMuByT(requestArray("float", plan[gpu_index].getOptionCount()));
            plan[gpu_index].setVBySqrtT(requestArray("float", plan[gpu_index].getOptionCount()));
            plan[gpu_index].setExpected(requestArray("float", plan[gpu_index].getOptionCount()));
            plan[gpu_index].setConfidence(requestArray("float", plan[gpu_index].getOptionCount()));

            // TODO: do these need to be Value as well?
            plan[gpu_index].setD(requestArray("long", plan[gpu_index].getGridSize() * THREAD_N));
            plan[gpu_index].setV(requestArray("long", plan[gpu_index].getGridSize() * THREAD_N * 5)); // v[5] is an array of 5 elements, hence gridSize*5*THREAD_N
            plan[gpu_index].setBoxmuller_flag(requestArray("int", plan[gpu_index].getGridSize() * THREAD_N));
            plan[gpu_index].setBoxmuller_flag_double(requestArray("int", plan[gpu_index].getGridSize() * THREAD_N));
            plan[gpu_index].setBoxmuller_extra(requestArray("float", plan[gpu_index].getGridSize() * THREAD_N));
            plan[gpu_index].setBoxmuller_extra_double(requestArray("double", plan[gpu_index].getGridSize() * THREAD_N));
        }

        Value cu = context.eval("grcuda", "CU");

        // TODO: possibly make this path relative
        String path_to_binded_kernels = PATH_TO_BINDED_FUNCTIONS + "kernels.ptx";
        //String path_to_binded_host_functions = PATH_TO_BINDED_FUNCTIONS + "libutils.so";
        checkFileExists(path_to_binded_kernels);
        //checkFileExists(path_to_binded_host_functions);

        rngSetupStatesKernelFunction = cu.invokeMember("bindkernel", path_to_binded_kernels, 
            "cxx rngSetupStates(" +
                "d: out pointer uint32, " +
                "v: out pointer uint32, " +
                "boxmuller_flag: out pointer sint32, " +
                "boxmuller_flag_double: out pointer sint32, " +
                "boxmuller_extra: out pointer float, " +
                "boxmuller_extra_double: out pointer double, " +
                "device_id: sint32)"
            );

        monteCarloOneBlockPerOptionKernelFunction = cu.invokeMember("bindkernel", path_to_binded_kernels,
            "cxx MonteCarloOneBlockPerOption( " +
                "d: out pointer uint32, " +
                "v: out pointer uint32, " +
                "boxmuller_flag: out pointer sint32, " +
                "boxmuller_flag_double: out pointer sint32, " +
                "boxmuller_extra: out pointer float, " +
                "boxmuller_extra_double: out pointer double, " +
                "optionData_S: out pointer float, " +
                "optionData_X: out pointer float, " +
                "optionData_MuByT: out pointer float, " +                
                "optionData_VBySqrtT: out pointer float, " +
                "callValue_Expected: out pointer float, " +
                "callValue_Confidence: out pointer float, " +
                "pathN: sint32, " +
                "optionN: sint32)"
            );
            
    }

    private void checkFileExists(String path_to_file) {
        File f = new File(path_to_file);
        if (!f.exists() || f.isDirectory())
            throw new RuntimeException("File doesn't exist");
    }

    /**
     * This method resembles the second part of initMonteCarloGPU.
     * It initializes the curandStates by launching the kernel rngSetupStates.
     */
    @Override
    protected void initializeTest(int iteration) { // TRANSLATION FINISHED!
        if (config.debug)
            System.out.println("    INSIDE initializeTest() - " + iteration);
        for (int gpu_index = 0; gpu_index < config.numGpus; gpu_index++)
            rngSetupStatesKernelFunction.execute(plan[gpu_index].getGridSize(), THREAD_N)
                    .execute(plan[gpu_index].getD(), plan[gpu_index].getV(), plan[gpu_index].getBoxmuller_flag(),
                            plan[gpu_index].getBoxmuller_flag_double(), plan[gpu_index].getBoxmuller_extra(),
                            plan[gpu_index].getBoxmuller_extra_double(), plan[gpu_index].getDevice());
    }

    @Override
    protected void resetIteration(int iteration) {
        //throw new UnsupportedOperationException("Unimplemented method 'resetIteration'");
    }

    @Override
    protected void runTest(int iteration) { // TRANSLATION FINISHED!
        for (int gpu_index = 0; gpu_index < config.numGpus; gpu_index++) {

            if (plan[gpu_index].getOptionCount() <= 0 || plan[gpu_index].getOptionCount() > Max_OPTIONS) {
                System.out.println("runTest(): bad option count");
                return;
            }

            double T, R, V, MuByT, VBySqrtT;

            for (int option_index = 0; option_index < plan[gpu_index].getOptionCount(); option_index++) {
                T = plan[gpu_index].getOptionData()[option_index].getT();
                R = plan[gpu_index].getOptionData()[option_index].getR();
                V = plan[gpu_index].getOptionData()[option_index].getV();
                MuByT = (R - (0.5 * V * V)) * T;
                VBySqrtT = V * Math.sqrt(T);

                plan[gpu_index].getS().setArrayElement(option_index,
                        plan[gpu_index].getOptionData()[option_index].getS());
                plan[gpu_index].getX().setArrayElement(option_index,
                        plan[gpu_index].getOptionData()[option_index].getX());
                plan[gpu_index].getMuByT().setArrayElement(option_index, (float)MuByT);
                plan[gpu_index].getVBySqrtT().setArrayElement(option_index, (float)VBySqrtT);
            }

            monteCarloOneBlockPerOptionKernelFunction.execute(plan[gpu_index].getGridSize(), THREAD_N)
                    .execute(
                            plan[gpu_index].getD(), 
                            plan[gpu_index].getV(),
                            plan[gpu_index].getBoxmuller_flag(),
                            plan[gpu_index].getBoxmuller_flag_double(),
                            plan[gpu_index].getBoxmuller_extra(),
                            plan[gpu_index].getBoxmuller_extra_double(),
                            plan[gpu_index].getS(),
                            plan[gpu_index].getX(),
                            plan[gpu_index].getMuByT(),
                            plan[gpu_index].getVBySqrtT(),
                            plan[gpu_index].getExpected(),
                            plan[gpu_index].getConfidence(),
                            plan[gpu_index].getPathN(),
                            plan[gpu_index].getOptionCount());
        }
    }

    @Override
    protected void closeTest(int iteration) { // TRANSLATION FINISHED!
        for (int gpu_index = 0; gpu_index < config.numGpus; gpu_index++) {
            for (int i = 0; i < plan[gpu_index].getOptionCount(); i++) {       
                double RT = plan[gpu_index].getOptionData()[i].getR() * plan[gpu_index].getOptionData()[i].getT();
                double sum = plan[gpu_index].getExpected().getArrayElement(i).asDouble();
                double sum2 = plan[gpu_index].getConfidence().getArrayElement(i).asDouble();
                double pathN = plan[gpu_index].getPathN();
                // Derive average from the total sum and discount by riskfree rate
                plan[gpu_index].setCallValueExpected(i, (float)(Math.exp(-RT) * sum / pathN));
                // Standard deviation
                double stdDev = Math.sqrt((pathN * sum2 - sum * sum) / (pathN * (pathN - 1)));
                // Confidence width; in 95% of all cases theoretical value lies within these borders
                plan[gpu_index].setCallValueConfidence(i, (float)(Math.exp(-RT) * 1.96 * stdDev / Math.sqrt(pathN)));
            }
        }
    }

    /**
     * This method execute Black-Scholes computation on the CPU and compares the
     * MonteCarlo results generated with the GPUs.
     */
    @Override
    protected void cpuValidation() { // TRANSLATION FINISHED!
        System.out.println(">>> Comparing Monte Carlo and Black-Scholes results...");
        //TODO: not feasible I guess
        //float[] callValueBS = new float[OPT_N];
        float callValueBS;
        float delta, ref;

        float sumDelta = 0;
        float sumRef = 0;
        float sumReserve = 0;

        // Iterate through all plans and compare BlackScholes results for each option
        for (int gpu_index = 0; gpu_index < config.numGpus; gpu_index++) {
            for (int option_index = 0; option_index < plan[gpu_index].getOptionCount(); option_index++) {
                callValueBS = Utils.BlackScholesCall(plan[gpu_index].getOptionData()[option_index]);
                delta = Math
                        .abs(callValueBS - plan[gpu_index].getCallValue()[option_index].getExpected());
                ref = callValueBS;
                sumDelta += delta;
                sumRef = Math.abs(ref);
                if (delta > 1e-6) {
                    sumReserve += plan[gpu_index].getCallValue()[option_index].getConfidence() / delta;
                }
            }
        }
        sumReserve /= OPT_N;
        
        System.out.println("Test Summary...");
        System.out.println("L1 norm        : " + (sumDelta / sumRef));
        System.out.println("Average reserve: " + sumReserve);
        if(sumReserve <= 1.0f)
            throw new RuntimeException("Test failed"); 
        //assertTrue("Test failed!", sumReserve > 1.0f);
    }

}