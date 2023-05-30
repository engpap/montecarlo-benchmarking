package com.polimi.montecarlo;

import org.graalvm.polyglot.Value;

public class Plan {
    // Device ID used for initialization of curandState
    int device;

    // Option count for this plan
    private int optionCount;

    // OptionData and OptionValue resembles optionData and callValue in MonteCarlo.cu
    private OptionData[] optionData;
    private OptionValue[] callValue;

    // OptionData and OptionValue resembles um_OptionData and um_CallValue in MonteCarlo_kernel.cu
    private OptionData_Value optionData_Value = new OptionData_Value();
    private OptionValue_Value optionValue_Value = new OptionValue_Value();

    private CurandState curandState = new CurandState();

    private int pathN;
    private int gridSize;

    /**
     * This class resembles the plan->um_OptionData struct in MonteCarlo_kernel.cu of a single plan element.
     */ 
    private class OptionData_Value { private Value S, X, MuByT, VBySqrtT; } 

    /**
     * This class resembles the plan->um_CallValue struct in MonteCarlo_kernel.cu of a single plan element.
     */ 
    private class OptionValue_Value { private Value Expected, Confidence; }

    /**
     * This class resembles the curandState struct, found in curand_kernel.h.
     * 
     * @implNote GrCUDA does not directly support unsigned integer data types, as Java itself doesn't have built-in support for unsigned integers.
     * However, we can work around this by using a larger data type. For example, we can use long to represent an unsigned int from C++.
     * This allows you to use the full range of unsigned int values from C++, since a long in Java has more bits than an int and thus can 
     * represent larger values. 
     * @implNote since rngStates is an array of curandState states, we decompose curandState struct into arrays of its components.
     */
    private class CurandState {
        private Value d;
        private Value v;
        private Value boxmuller_flag;
        private Value boxmuller_flag_double;
        private Value boxmuller_extra;
        private Value boxmuller_extra_double;
    }

    /// CONSTRUCTOR
    public Plan(int optionCount) {
        // Initialize the optionData and optionValue arrays
        this.optionCount = optionCount;
        this.optionData = new OptionData[optionCount];
        this.callValue = new OptionValue[optionCount];
    }

    /// SETTERS

    public void setDevice(int device) {
        this.device = device;
    }

    public void setOptionCount(int optionCount) {
        this.optionCount = optionCount;
    }

    /**
     * Initializes the plan's optionData array, copying it from the array passed as input, from
     * index l to index r.
     * 
     * @param optionData the option array.
     * @param l the index of the first option to copy.
     * @param r the index of the option after the last option to copy.
     */
    public void setOptionData(OptionData[] optionData, int l, int r) {
        int c = 0;
        for (int i = l; i < r; i++) {
            this.optionData[c] = optionData[i];
            c++;
        }
    }

    /**
     * Initializes the plan's callValue array, copying it from the array passed as input, from
     * index l to index r.
     * 
     * @param callValue the call value array.
     * @param l the index of the first call value to copy.
     * @param r the index of the call value after the last option to copy.
     */
    public void setCallValue(OptionValue[] callValue, int l, int r) {
        int c = 0;
        for (int i = l; i < r; i++) {
            this.callValue[c] = callValue[i];
            c++;
        }
    }

    public void setCallValueExpected(int index, float Expected){
        this.callValue[index].setExpected(Expected);
    }
    
    public void setCallValueConfidence(int index, float Confidence){
        this.callValue[index].setConfidence(Confidence);
    }


    public void setS(Value S) {
        this.optionData_Value.S = S;
    }

    public void setX(Value X) {
        this.optionData_Value.X = X;
    }

    public void setMuByT(Value MuByT) {
        this.optionData_Value.MuByT = MuByT;
    }

    public void setVBySqrtT(Value VBySqrtT) {
        this.optionData_Value.VBySqrtT = VBySqrtT;
    }

    public void setExpected(Value Expected) {
        this.optionValue_Value.Expected = Expected;
    }

    public void setConfidence(Value Confidence) {
        this.optionValue_Value.Confidence = Confidence;
    }
    
    public void setD(Value d) {
        this.curandState.d = d;
    }

    public void setV(Value v) {
        this.curandState.v = v;
    }

    public void setBoxmuller_flag(Value boxmuller_flag) {
        this.curandState.boxmuller_flag = boxmuller_flag;
    }

    public void setBoxmuller_flag_double(Value boxmuller_flag_double) {
        this.curandState.boxmuller_flag_double = boxmuller_flag_double;
    }

    public void setBoxmuller_extra(Value boxmuller_extra) {
        this.curandState.boxmuller_extra = boxmuller_extra;
    }

    public void setBoxmuller_extra_double(Value boxmuller_extra_double) {
        this.curandState.boxmuller_extra_double = boxmuller_extra_double;
    }

    public void setPathN(int pathN) {
        this.pathN = pathN;
    }

    public void setGridSize(int gridSize) {
        this.gridSize = gridSize;
    }

    /// GETTERS

    public int getDevice() {
        return this.device;
    }

    public int getOptionCount() {
        return this.optionCount;
    }

    public Value getS() {
        return this.optionData_Value.S;
    }

    public Value getX() {
        return this.optionData_Value.X;
    }

    public Value getMuByT() {
        return this.optionData_Value.MuByT;
    }

    public Value getVBySqrtT() {
        return this.optionData_Value.VBySqrtT;
    }

    public Value getExpected() {
        return this.optionValue_Value.Expected;
    }

    public Value getConfidence() {
        return this.optionValue_Value.Confidence;
    }

    public Value getD() {
        return this.curandState.d;
    }

    public Value getV() {
        return this.curandState.v;
    }

    public Value getBoxmuller_flag() {
        return this.curandState.boxmuller_flag;
    }

    public Value getBoxmuller_flag_double() {
        return this.curandState.boxmuller_flag_double;
    }

    public Value getBoxmuller_extra() {
        return this.curandState.boxmuller_extra;
    }

    public Value getBoxmuller_extra_double() {
        return this.curandState.boxmuller_extra_double;
    }

    public int getPathN() {
        return this.pathN;
    }

    public OptionData[] getOptionData() {
        return this.optionData;
    }

    public OptionValue[] getCallValue() {
        return this.callValue;
    }

    public int getGridSize() {
        return this.gridSize;
    }
}


