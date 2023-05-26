package com.polimi.montecarlo;

public class OptionData {
    private float S, X, T, R, V;

    public OptionData(float S, float X, float T, float R, float V) {
        this.S = S;
        this.X = X;
        this.T = T;
        this.R = R;
        this.V = V;
    }

    public float getS() {
        return S;
    }

    public float getX() {
        return X;
    }

    public float getT() {
        return T;
    }

    public float getR() {
        return R;
    }

    public float getV() {
        return V;
    }
}
