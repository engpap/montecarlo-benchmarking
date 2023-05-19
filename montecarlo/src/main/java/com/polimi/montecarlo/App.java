package com.polimi.montecarlo;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        BenchmarkConfig config = new BenchmarkConfig();
        config.benchmarkName = "B1";
        config.setupId = "B1";
        Benchmark b1 = new B1(new BenchmarkConfig());
        b1.run();
    }
}
