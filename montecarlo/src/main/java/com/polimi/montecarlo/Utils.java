package com.polimi.montecarlo;

import java.util.Random;

public class Utils {
    public static float BlackScholesCall(OptionData optionData) {

        double S = optionData.getS();
        double X = optionData.getX();
        double T = optionData.getT();
        double R = optionData.getR();
        double V = optionData.getV();

        double sqrtT = Math.sqrt(T);
        double d1 = (Math.log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
        double d2 = d1 - V * sqrtT;
        double CNDD1 = CND(d1);
        double CNDD2 = CND(d2);
        double expRT = Math.exp(- R * T);

        return (float)(S * CNDD1 - (X * expRT * CNDD2));
    }

    private static double CND(double d) {
        // Black-Scholes parameters
        double A1 = 0.31938153;
        double A2 = -0.356563782;
        double A3 = 1.781477937;
        double A4 = -1.821255978;
        double A5 = 1.330274429;
        double RSQRT2PI = 0.39894228040143267793994605993438;

        double K = 1.0 / (1.0 + 0.2316419 * Math.abs(d));

        double cnd = RSQRT2PI * Math.exp((-0.5 * d * d)) *
            (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

        if (d > 0) {
            cnd = 1.0 - cnd;
        }

        return cnd;
    }
    
    public static float randFloat(float low, float high) {
        Random random = new Random();
        float t = random.nextFloat();
        return (1.0f - t) * low + t * high;
    }
}
