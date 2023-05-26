package com.polimi.montecarlo;


import com.fasterxml.jackson.annotation.JsonIgnore;

/**
 * This class will be passed to initialize the configuration of a benchmark.
 */
public class BenchmarkConfig {
    /**
     * Default parameters
     */
    public String benchmarkName = "";
    public String setupId = "";
    public int totIter;
    public int currentIter;
    public int randomSeed = 42;
    public int size;
    public int blockSize1D = 32;
    public int blockSize2D = 8;
    boolean timePhases = false;
    public int numBlocks = 8;
    public boolean randomInit = false;
    public boolean reInit = false;
    public boolean reAlloc = false;
    // TODO: mbhhhh
    public boolean cpuValidate = true;
    // GrCUDA context settings
    public String executionPolicy;
    public boolean inputPrefetch;
    public String retrieveNewStreamPolicy;
    public String retrieveParentStreamPolicy;
    public String dependencyPolicy;
    public String deviceSelectionPolicy;
    public boolean forceStreamAttach;
    public boolean enableComputationTimers;
    public int numGpus;
    // VALUES ADDED FOR MC //
    public int deviceId;
    public int pathN;
    public int optN;
    // VALUES ADDED FOR MC //
    public String memAdvisePolicy;
    public String scalingChoice;
    @JsonIgnore public String bandwidthMatrix;
    // Debug parameters
    public boolean debug;
    public boolean nvprof_profile;
    public String gpuModel;
    @JsonIgnore public String results_path;

    @Override
    public String toString() {
        return "BenchmarkConfig{" +
                "benchmarkName='" + benchmarkName + '\'' +
                ", setupId='" + setupId + '\'' +
                ", totIter=" + totIter +
                ", currentIter=" + currentIter +
                ", randomSeed=" + randomSeed +
                ", size=" + size +
                ", blockSize1D=" + blockSize1D +
                ", blockSize2D=" + blockSize2D +
                ", timePhases=" + timePhases +
                ", numBlocks=" + numBlocks +
                ", randomInit=" + randomInit +
                ", reInit=" + reInit +
                ", reAlloc=" +reAlloc+
                ", cpuValidate=" + cpuValidate +
                ", executionPolicy='" + executionPolicy + '\'' +
                ", inputPrefetch=" + inputPrefetch +
                ", retrieveNewStreamPolicy='" + retrieveNewStreamPolicy + '\'' +
                ", retrieveParentStreamPolicy='" + retrieveParentStreamPolicy + '\'' +
                ", dependencyPolicy='" + dependencyPolicy + '\'' +
                ", deviceSelectionPolicy='" + deviceSelectionPolicy + '\'' +
                ", forceStreamAttach=" + forceStreamAttach +
                ", enableComputationTimers=" + enableComputationTimers +
                ", numGpus=" + numGpus +
                ", memAdvisePolicy='" + memAdvisePolicy + '\'' +
                ", scalingChoice='" + scalingChoice + '\'' +
                ", bandwidthMatrix='" + bandwidthMatrix + '\'' +
                '}';
    }
}