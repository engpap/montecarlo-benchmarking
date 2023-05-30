package com.polimi.montecarlo;

import java.util.Random;

/**
 * This class contains some functions utilized by the MonteCarlo benchmark class.
 */
public class Utils {

    /**
     * Computes the call value for the option passed as input using the Black-Scholes formula.
     * @see https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#Black%E2%80%93Scholes_formula
     * 
     * @param optionData the option.
     * @return the Black-Scholes result for the current option.
     */
    public static float BlackScholesCall(OptionData optionData) {
        // get the option data
        double S = optionData.getS();
        double X = optionData.getX();
        double T = optionData.getT();
        double R = optionData.getR();
        double V = optionData.getV();
        // Compute the parameters required for the formula
        double sqrtT = Math.sqrt(T);
        double d1 = (Math.log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
        double d2 = d1 - V * sqrtT;
        double CNDD1 = CND(d1);
        double CNDD2 = CND(d2);
        double expRT = Math.exp(- R * T);
        // Return the call value
        return (float)(S * CNDD1 - X * expRT * CNDD2);
    }

    /**
     * Computes the Cumulative Normal Distribution, required by the Black-Scholes formula.
     * 
     * @param d respectively d1 or d2
     * @return the cumulative distribution.
     */
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
    
    /**
     * Returns a random float between the range passed as input.
     * 
     * @param low the lower bound of the range.
     * @param high the upper bound of the range.
     * 
     * @return a random float between low and high.
     * 
     * @implNote For consistency, the seed is fixed to 123 as the cpp MonteCarlo benchmarking version.
     * Note that generated random numbers are different from the cpp version, even if the same seed is used.
     */
    public static float randFloat(float low, float high) {
        Random random = new Random(123);
        float t = random.nextFloat();
        return (1.0f - t) * low + t * high;
    }
}
