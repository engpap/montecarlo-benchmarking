package com.polimi.montecarlo;

import java.io.File;

import static org.junit.Assert.assertTrue;

import org.graalvm.polyglot.Value;

import java.nio.file.Paths;

public class MonteCarlo extends Benchmark {

    // Dynamically get the path of cpp_libs folder
    private static final String PATH_TO_BINDED_FUNCTIONS = Paths.get("").toAbsolutePath().toString() + "/cpp_libs/";

    /// Kernels
    private Value monteCarloOneBlockPerOptionKernelFunction;
    private Value rngSetupStatesKernelFunction;

    /// Host Functions
    private Value adjustProblemSizeFunction;
    private Value adjustGridSizeFunction;

    /// Constants
    private static final int THREAD_N = 256;
    private static final int Max_OPTIONS = 1024 * 1024 * 64;

    // Total number of options in the current benchmark run
    private int OPT_N;

    /// OptionData and CallValueGPU are arrays of OPT_N elements
    private OptionData[] optionData;
    private OptionValue[] callValueGPU;

    // OptionPlan struct equivalent, array of config.numGpus elements
    private Plan[] plan;

    // Class constructor to initialize the benchmark based on the configuration parameters
    public MonteCarlo(BenchmarkConfig currentConfig) {
        super(currentConfig);
        init();
    }

    /**
     * @montecarlo This method resembles the implementation of intialization functions of
     * MonteCarloMultiGPU.cpp
     */
    private void init() {
        // Initialize c++ host functions, not available in Java
        setupAdjustSizeHostFunctions();
        // Initialize kernel functions for the benchmark
        setupKernelFunctions(); 
        // Number of options initialization
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
        nOptions = adjustProblemSizeFunction.execute(config.numGpus, nOptions).asInt();
        int scale = (strongScaling) ? 1 : config.numGpus;
        OPT_N = nOptions * scale;
        config.optN = OPT_N; // useful for printing out results on .csv file
        int PATH_N = 262144;
        config.pathN = PATH_N; // useful for printing out results on .csv file

        // Instantiate the optionData and OptionValue arrays
        optionData = new OptionData[OPT_N];
        callValueGPU = new OptionValue[OPT_N];
        float S, X, T, R, V;
        float Expected, Confidence;
        // Initialize each option's parameters
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
            // Take into account cases with "odd" option counts
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

        // Assign GPU option and callValue ranges to each Plan, as well as other parameters required by the benchmark
        int gpuBase = 0;
        for (int gpu_index = 0; gpu_index < config.numGpus; gpu_index++) {
            plan[gpu_index].setDevice(gpu_index);
            plan[gpu_index].setOptionData(optionData, gpuBase, gpuBase + plan[gpu_index].getOptionCount());
            plan[gpu_index].setCallValue(callValueGPU, gpuBase, gpuBase + plan[gpu_index].getOptionCount());
            plan[gpu_index].setPathN(PATH_N);
            plan[gpu_index].setGridSize(adjustGridSizeFunction.execute(plan[gpu_index].getDevice(), plan[gpu_index].getOptionCount()).asInt());
            gpuBase += plan[gpu_index].getOptionCount();
        }
    }

    /**
     * Retrieves the path to the file where the custom host functions are written and binds 
     * them to the respective Value instances.   
     */
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
     * Retrieves the path to the file where the custom kernel functions are written and binds
     * them to the respective Value instances.
     */
    private void setupKernelFunctions() {
        //TODO: find a better way for this
        String GPU = config.gpuModel.contains("P100") ? "p100" : "v100";
        String path_to_binded_kernels = PATH_TO_BINDED_FUNCTIONS + "kernels_" + GPU + ".ptx";
        Value cu = context.eval("grcuda", "CU");
        checkFileExists(path_to_binded_kernels);

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

    /**
     * @montecarlo This method resembles the first part of initMonteCarloGPU.
     * It allocates memory for all the arrays used in the benchmark.
     */
    @Override
    protected void allocateTest(int iteration) {
        // Allocate memory for optionData and optionValue (callValue)
        for (int gpu_index = 0; gpu_index < config.numGpus; gpu_index++) {
            plan[gpu_index].setS(requestArray("float", plan[gpu_index].getOptionCount()));
            plan[gpu_index].setX(requestArray("float", plan[gpu_index].getOptionCount()));
            plan[gpu_index].setMuByT(requestArray("float", plan[gpu_index].getOptionCount()));
            plan[gpu_index].setVBySqrtT(requestArray("float", plan[gpu_index].getOptionCount()));
            plan[gpu_index].setExpected(requestArray("float", plan[gpu_index].getOptionCount()));
            plan[gpu_index].setConfidence(requestArray("float", plan[gpu_index].getOptionCount()));
        // Allocate memory for the curandStates0
            plan[gpu_index].setD(requestArray("long", plan[gpu_index].getGridSize() * THREAD_N));
            plan[gpu_index].setV(requestArray("long", plan[gpu_index].getGridSize() * THREAD_N * 5)); // v[5] is an array of 5 elements, hence gridSize*5*THREAD_N
            plan[gpu_index].setBoxmuller_flag(requestArray("int", plan[gpu_index].getGridSize() * THREAD_N));
            plan[gpu_index].setBoxmuller_flag_double(requestArray("int", plan[gpu_index].getGridSize() * THREAD_N));
            plan[gpu_index].setBoxmuller_extra(requestArray("float", plan[gpu_index].getGridSize() * THREAD_N));
            plan[gpu_index].setBoxmuller_extra_double(requestArray("double", plan[gpu_index].getGridSize() * THREAD_N));
        } 
    }

    /**
     * Tries to open the file using the path passed as input.
     * 
     * @param path_to_file the path to the file.
     * 
     * @throws RuntimeException if it is unable to open the requested file.
     */
    private void checkFileExists(String path_to_file) {
        File f = new File(path_to_file);
        if (!f.exists() || f.isDirectory())
            throw new RuntimeException("File doesn't exist");
    }

    /**
     * @montecarlo This method resembles the second part of initMonteCarloGPU.
     * It initializes the curandStates by launching the kernel rngSetupStates.
     */
    @Override
    protected void initializeTest(int iteration) {
        if (config.debug)
            System.out.println("    INSIDE initializeTest() - " + iteration);
        // for each GPU execute the rngSetupStates kernel
        for (int gpu_index = 0; gpu_index < config.numGpus; gpu_index++)
            rngSetupStatesKernelFunction.execute(plan[gpu_index].getGridSize(), THREAD_N)
                    .execute(plan[gpu_index].getD(), plan[gpu_index].getV(), plan[gpu_index].getBoxmuller_flag(),
                            plan[gpu_index].getBoxmuller_flag_double(), plan[gpu_index].getBoxmuller_extra(),
                            plan[gpu_index].getBoxmuller_extra_double(), plan[gpu_index].getDevice());
    }

    /**
     * Not used
     */
    @Override
    protected void resetIteration(int iteration) {
    }

    /**
     * @montecarlo This method resembles the whole MonteCarloGPU function.
     * It moves the option data in the Value arrays and calls the main kernel.
     */
    @Override
    protected void runTest(int iteration) {
        // For each GPU plan (optionSolver)
        for (int gpu_index = 0; gpu_index < config.numGpus; gpu_index++) {

            // check if the number of options is valid
            if (plan[gpu_index].getOptionCount() <= 0 || plan[gpu_index].getOptionCount() > Max_OPTIONS) {
                System.out.println("runTest(): bad option count");
                return;
            }

            double T, R, V, MuByT, VBySqrtT;
            // For each option
            for (int option_index = 0; option_index < plan[gpu_index].getOptionCount(); option_index++) {
                // compute the MuByT and the VbySqrtT values
                T = plan[gpu_index].getOptionData()[option_index].getT();
                R = plan[gpu_index].getOptionData()[option_index].getR();
                V = plan[gpu_index].getOptionData()[option_index].getV();
                MuByT = (R - (0.5 * V * V)) * T;
                VBySqrtT = V * Math.sqrt(T);
                // Copy the required data inside the Value array
                plan[gpu_index].getS().setArrayElement(option_index,
                        plan[gpu_index].getOptionData()[option_index].getS());
                plan[gpu_index].getX().setArrayElement(option_index,
                        plan[gpu_index].getOptionData()[option_index].getX());
                plan[gpu_index].getMuByT().setArrayElement(option_index, (float)MuByT);
                plan[gpu_index].getVBySqrtT().setArrayElement(option_index, (float)VBySqrtT);
            }

            // Execute the MonteCarloOneBlockPerOption kernel
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

            context.eval("grcuda", "cudaMemPrefetchAsync").execute();
        }
    }

    /**
     * @montecarlo This method resembles the whole closeMonteCarloGPU function.
     * It copies the data from the Value arrays to the callValue arrays.
     */
    @Override
    protected void closeTest(int iteration) { 
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
     * @montecarlo This method resembles the final check at the end of MonteCarloMultiGpu.cpp.
     * It executes the Black-Scholes computation on the CPU and compares it with the
     * MonteCarlo results generated by the GPUs.
     * 
     * TODO: add the throws clause in the documentation 
     */
    @Override
    protected void cpuValidation() {
        System.out.println(">>> Comparing Monte Carlo and Black-Scholes results...");
        //TODO: not feasible I guess
        //float[] callValueBS = new float[OPT_N];
        // Black-Scholes result
        float callValueBS;
        float delta, ref;

        float sumDelta = 0;
        float sumRef = 0;
        float sumReserve = 0;

        // For each GPU plan (optionSolver)
        for (int gpu_index = 0; gpu_index < config.numGpus; gpu_index++) {
            // For each option of the plan
            for (int option_index = 0; option_index < plan[gpu_index].getOptionCount(); option_index++) {
                // Compute the Black-Scholes result
                callValueBS = Utils.BlackScholesCall(plan[gpu_index].getOptionData()[option_index]);
                // Compute the delta between the two results
                delta = Math
                        .abs(callValueBS - plan[gpu_index].getCallValue()[option_index].getExpected());
                // Increase the sumDelta and the sumRef values
                ref = callValueBS;
                sumDelta += delta;
                sumRef += Math.abs(ref);
                // Increase the reserve if the delta is bigger than 1e-6
                if (delta > 1e-6) {
                    sumReserve += plan[gpu_index].getCallValue()[option_index].getConfidence() / delta;
                }
            }
        }
        // Divide the sumReserve by the number of options
        sumReserve /= OPT_N;
        
        System.out.println("Test Summary...");
        System.out.printf("L1 norm        : %E%n", sumDelta / sumRef);
        System.out.println("Average reserve: " + sumReserve);
        if(sumReserve <= 1.0f)
            throw new RuntimeException("Test failed"); 
        assertTrue("Test Failed", sumReserve > 1.0f);
    }

}