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

    /// This class resembles the plan->um_OptionData struct in MonteCarlo_kernel.cu of a single plan element
    private class OptionData_Value { private Value S, X, MuByT, VBySqrtT; } 

    /// This class resembles the plan->um_CallValue struct in MonteCarlo_kernel.cu of a single plan element
    private class OptionValue_Value { private Value Expected, Confidence; }

    private int pathN;

    private int gridSize;

    /// SETTERS

    public void setDevice(int device) {
        this.device = device;
    }

    public void setOptionCount(int optionCount) {
        this.optionCount = optionCount;
    }

    public void setOptionData(OptionData[] optionData, int l, int r) {
        for (int i = l; i < r; i++) {
            this.optionData[i] = optionData[i];
        }
    }

    public void setCallValue(OptionValue[] callValue, int l, int r) {
        for (int i = l; i < r; i++) {
            this.callValue[i] = callValue[i];
        }
    }

    public void setS(Value S) {
        this.optionData_Value.S = S;
    }

    public void setX(Value X) {
        this.optionData_Value.X = X;
    }

    public void setMuByT(Value T) {
        this.optionData_Value.X = T;
    }

    public void setVBySqrtT(Value R) {
        this.optionData_Value.VBySqrtT = R;
    }

    public void setExpected(Value Expected) {
        this.optionValue_Value.Expected = Expected;
    }

    public void setConfidence(Value Confidence) {
        this.optionValue_Value.Confidence = Confidence;
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

    public int getPATH_N() {
        return this.pathN;
    }

    public OptionData[] getOptionData() {
        return this.optionData;
    }

    public OptionValue[] getCallValue() {
        return this.callValue;
    }
    
}


