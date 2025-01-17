package com.polimi.montecarlo;


import java.util.ArrayList;
import java.util.function.Consumer;

//import org.graalvm.compiler.core.common.type.SymbolicJVMCIReference;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;


public abstract class Benchmark {
    public Context context;
    public final BenchmarkConfig config;
    public final BenchmarkResults benchmarkResults;
    public ArrayList<Value> deviceArrayList = new ArrayList<>(); // used to store all the arrays to be freed at the end of the benchmark

    public Benchmark(BenchmarkConfig currentConfig) {
        this.config = currentConfig;
        this.benchmarkResults = new BenchmarkResults(currentConfig);
        this.context = createContext(currentConfig);
    }

    /**
     * This method is used to run the current benchmark.
     * It will use the information stored in the config attribute to decide whether to do an additional initialization phase and
     the cpuValidation.
     */
    public void run() {
        if(config.debug)
            System.out.println("INSIDE run()");

        for (int i = 0; i < config.totIter; ++i) {
            if(config.debug)
                System.out.println("["+i+"] START");
            benchmarkResults.startNewIteration(i, config.timePhases); // create the current iteration in the result class

            // Start a timer to monitor the total GPU execution time
            long overall_gpu_start = System.nanoTime();

            // Allocate memory for the benchmark
    
            if(config.nvprof_profile){
                context.eval("grcuda", "cudaProfilerStart").execute();
            }            

            if (config.reAlloc || i == 0){
                if(config.debug)
                    System.out.println("["+i+"] alloc");
                time(i, "alloc", this::allocateTest);
            }

            // Initialize memory for the benchmark

            if (config.reInit || i == 0){
                if(config.debug)
                    System.out.println("["+i+"] init");
                time(i, "init", this::initializeTest);
            }

            /* // Reset the result
            if(config.debug)
                System.out.println("["+i+"] reset");
            time(i, "reset", this::resetIteration);*/
            
            // Execute the benchmark
            if(config.debug)
                System.out.println("["+i+"] execution");
            time(i, "execution", this::runTest);

            if(config.debug)
                System.out.println("["+i+"] close");
            time(i, "close", this::closeTest);

            if(config.nvprof_profile){
                context.eval("grcuda", "cudaProfilerStop").execute();
            }

            // Stop the timer
            long overall_gpu_end = System.nanoTime();

            benchmarkResults.setCurrentTotalTime((overall_gpu_end - overall_gpu_start) / 1000000000F);

            // Perform validation on CPU
            if (config.cpuValidate && i == 0) {
                cpuValidation();
            }

            /* NOT RELEVANT FOR THIS BENCHMARK BECAUSE IT COMPUTES THOUSANDS OF OPTIONS
            if(config.debug)
               System.out.println("["+i+"] VALIDATION \nCPU: " + benchmarkResults.cpu_result+"\nGPU: " + benchmarkResults.currentIteration().gpu_result);
            */

            // At every iteration, save the results to a csv file
            benchmarkResults.saveToCsvFile();
        }

        // Save the benchmark results
        //benchmarkResults.saveToJsonFile();

        // Free the allocated arrays
        deallocDeviceArrays();

        //  Gracefully close the current context
        context.close();
    }

    /**
     * This method is used to time the function passed to it.
     * It will add the timing and the phase name to the benchmarkResult attribute.
     * System.nanoTime method can only be used to measure elapsed time and is not related to any other notion of system or wall-clock time. 
     * @param iteration the current iteration of the benchmark
     * @param phaseName the current phase of the benchmark
     * @param functionToTime the function to time passed like "class::funName"
     */
    private void time(int iteration, String phaseName, Consumer<Integer> functionToTime){
        long begin = System.nanoTime();
        //long begin = System.currentTimeMillis();
        functionToTime.accept(iteration);
        long end = System.nanoTime();
        //long end = System.currentTimeMillis();
        benchmarkResults.addPhaseToCurrentIteration(phaseName, (end - begin)/ 1000000000F); // to sec
        //benchmarkResults.addPhaseToCurrentIteration(phaseName, (end - begin));
    }

    protected void deallocDeviceArrays(){
        for(Value v : deviceArrayList)
            v.invokeMember("free");
    }

    protected Value requestArray(String type, int size){
        Value vector = context.eval("grcuda", type+"["+ size +"]");
        deviceArrayList.add(vector);
        return vector;
    }

    private Context createContext(BenchmarkConfig config){
        return Context
                .newBuilder()
                .allowAllAccess(true)
                .allowExperimentalOptions(true)
                //logging settings
                .option("log.grcuda.com.nvidia.grcuda.level", "WARNING")
                .option("log.grcuda.com.nvidia.grcuda.GrCUDAContext.level", "SEVERE")
                //GrCUDA env settings
                .option("grcuda.ExecutionPolicy", config.executionPolicy)
                .option("grcuda.InputPrefetch", String.valueOf(config.inputPrefetch))
                .option("grcuda.RetrieveNewStreamPolicy", config.retrieveNewStreamPolicy)
                .option("grcuda.RetrieveParentStreamPolicy", config.retrieveParentStreamPolicy)
                .option("grcuda.DependencyPolicy", config.dependencyPolicy)
                .option("grcuda.DeviceSelectionPolicy", config.deviceSelectionPolicy)
                .option("grcuda.ForceStreamAttach", String.valueOf(config.forceStreamAttach))
                .option("grcuda.EnableComputationTimers", String.valueOf(config.enableComputationTimers))
                .option("grcuda.MemAdvisePolicy", config.memAdvisePolicy)
                .option("grcuda.NumberOfGPUs", String.valueOf(config.numGpus))
                .option("grcuda.BandwidthMatrix", config.bandwidthMatrix)
                .build();
    }

    /*
        ###################################################################################
                        METHODS TO BE IMPLEMENTED IN THE BENCHMARKS
        ###################################################################################
    */

    /**
     * Here goes the read of the test parameters,
     * the initialization of the necessary arrays
     * and the creation of the kernels (if applicable)
     * @param iteration the current number of the iteration
     */
    protected abstract void initializeTest(int iteration);

    /**
     * Allocate new memory on GPU used for the benchmark
     * @param iteration the current number of the iteration
     */
    protected abstract void allocateTest(int iteration);

    /**
     * Reset code, to be run before each test
     * Here you clean up the arrays and other reset stuffs
     * @param iteration the current number of the iteration
     */
    protected abstract void resetIteration(int iteration);

    /**
     * Run the actual test
     * @param iteration the current number of the iteration
     */
    protected abstract void runTest(int iteration);

    /**
     * Close the test computation
     * @param iteration the current number of the iteration
     */
    protected abstract void closeTest(int iteration);

    /**
     * (numerically) validate results against CPU
     */
    protected abstract void cpuValidation();

}