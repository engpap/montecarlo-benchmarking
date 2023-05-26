package com.polimi.montecarlo;

public class OptionValue {
    private float Expected, Confidence;

    public OptionValue(float Expected, float Confidence) {
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