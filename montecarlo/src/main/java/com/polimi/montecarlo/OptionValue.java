package com.polimi.montecarlo;

public class OptionValue {
    private float Expected, Confidence;

    public OptionValue(float Expected, float Confidence) {
       this.Expected = Expected;
       this.Confidence = Confidence;
    }

    /// GETTERS

    public float getExpected() {
        return Expected;
    }

    public float getConfidence() {
        return Confidence;
    }

    /// SETTERS

    public void setExpected(float Expected) {
        this.Expected = Expected;
    }

    public void setConfidence(float Confidence) {
        this.Confidence = Confidence;
    }
}