package com.polimi.montecarlo;
import com.polimi.montecarlo.B1;
import com.polimi.montecarlo.BenchmarkConfig;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        Benchmark b1 = new B1(new BenchmarkConfig());
        b1.run();
    }
}
