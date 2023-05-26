package com.polimi.montecarlo;

public class CallValueGPU {
    private float Expected, Confidence;

    public CallValueGPU(float Expected, float Confidence) {
       this.Expected = Expected;
       this.Confidence = Confidence;
    }

    public float getExpected() {
        return Expected;
    }

    public float getConfidence() {
        return Confidence;
    }
}