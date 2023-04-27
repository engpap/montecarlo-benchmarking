# Starting point

./MonteCarlo Starting...

Using single CPU thread for multiple GPUs
==35302== NVPROF is profiling process 35302, command: ./MonteCarlo --method=streamed --scaling=weak
MonteCarloMultiGPU
==================
Parallelization method  = streamed
Problem scaling         = weak
Number of GPUs          = 2
Total number of options = 209714
Number of paths         = 262144
main(): generating input data...
main(): starting 2 host threads...
main(): GPU statistics, streamed
GPU Device #0: Tesla V100-SXM2-16GB
Options         : 104857
Simulation paths: 262144
GPU Device #1: Tesla V100-SXM2-16GB
Options         : 104857
Simulation paths: 262144

Total time (ms.): 122.557999
        Note: This is elapsed time for all to compute.
Options per sec.: 1711140.866346
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.798536E-04
Average reserve: 12.047607

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed

>>> Inside solverThread, initMonteCarloGPU took 25.3328 milliseconds
>>> Inside solverThread, MonteCarloGPU took 122.557 milliseconds
==35302== Profiling application: ./MonteCarlo --method=streamed --scaling=weak
==35302== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   97.66%  238.08ms         2  119.04ms  119.03ms  119.05ms  MonteCarloOneBlockPerOption(curandStateXORWOW*, __TOptionData const *, __TOptionValue*, int, int)
                    2.34%  5.6975ms         2  2.8488ms  2.8412ms  2.8563ms  rngSetupStates(curandStateXORWOW*, int)
      API calls:   68.49%  333.32ms         2  166.66ms  159.98ms  173.34ms  cudaStreamCreate
                   24.49%  119.18ms         2  59.590ms  1.6561ms  117.52ms  cudaEventSynchronize
                    4.26%  20.753ms         6  3.4588ms  24.190us  20.518ms  cudaMallocManaged
                    0.84%  4.0893ms         6  681.54us  181.23us  1.5719ms  cudaFree
                    0.80%  3.9146ms        10  391.46us  29.948us  640.72us  cudaMemPrefetchAsync
                    0.58%  2.8434ms         2  1.4217ms  964.93us  1.8785ms  cudaDeviceSynchronize
                    0.37%  1.7922ms         8  224.02us  199.17us  279.24us  cudaGetDeviceProperties
                    0.10%  481.69us       202  2.3840us     173ns  115.19us  cuDeviceGetAttribute
                    0.02%  98.887us         4  24.721us  18.729us  34.046us  cudaLaunchKernel
                    0.01%  54.697us         2  27.348us  20.793us  33.904us  cuDeviceGetName
                    0.01%  32.786us        14  2.3410us     476ns  9.6440us  cudaSetDevice
                    0.01%  30.265us         2  15.132us  11.427us  18.838us  cudaStreamDestroy
                    0.00%  17.183us         2  8.5910us  2.7600us  14.423us  cuDeviceGetPCIBusId
                    0.00%  10.088us         2  5.0440us  4.0290us  6.0590us  cudaEventRecord
                    0.00%  6.2180us         2  3.1090us  2.9240us  3.2940us  cudaEventCreate
                    0.00%  4.0370us         2  2.0180us  1.8540us  2.1830us  cudaEventDestroy
                    0.00%  1.8740us         4     468ns     172ns     769ns  cuDeviceGet
                    0.00%  1.6080us         4     402ns     295ns     593ns  cudaGetLastError
                    0.00%  1.4500us         3     483ns     251ns     926ns  cuDeviceGetCount
                    0.00%  1.0870us         2     543ns     390ns     697ns  cuDeviceTotalMem
                    0.00%     990ns         1     990ns     990ns     990ns  cudaGetDeviceCount
                    0.00%     518ns         2     259ns     225ns     293ns  cuDeviceGetUuid
                    0.00%     419ns         1     419ns     419ns     419ns  cuModuleGetLoadingMode

==35302== Unified Memory profiling result:
Device "Tesla V100-SXM2-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
       1  1.6016MB  1.6016MB  1.6016MB  1.601563MB  175.6180us  Host To Device
       1  820.00KB  820.00KB  820.00KB  820.0000KB  65.60100us  Device To Host
Device "Tesla V100-SXM2-16GB (1)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
       1  1.6016MB  1.6016MB  1.6016MB  1.601563MB  165.2820us  Host To Device
       1  820.00KB  820.00KB  820.00KB  820.0000KB  75.81000us  Device To Host



# Before rngSetupStates kernel call

==37768== Profiling application: ./MonteCarlo --method=streamed --scaling=weak
==37768== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.76%  2.08419s         2  1.04209s  1.03478s  1.04941s  MonteCarloOneBlockPerOption(curandStateXORWOW*, __TOptionData const *, __TOptionValue*, int, int)
                    0.24%  5.0006ms         2  2.5003ms  2.4843ms  2.5163ms  rngSetupStates(curandStateXORWOW*, int)
      API calls:   73.71%  1.04956s         2  524.78ms  29.695ms  1.01987s  cudaEventSynchronize
                   22.04%  313.76ms         2  156.88ms  146.45ms  167.31ms  cudaStreamCreate
                    1.75%  24.969ms        10  2.4969ms  70.836us  5.8761ms  cudaMemPrefetchAsync
                    1.46%  20.735ms         6  3.4558ms  21.386us  20.524ms  cudaMallocManaged
                    0.66%  9.4627ms         6  1.5771ms  1.0599ms  2.1902ms  cudaFree
                    0.18%  2.5187ms         2  1.2594ms  936.97us  1.5817ms  cudaDeviceSynchronize
                    0.13%  1.8837ms         8  235.46us  196.76us  271.98us  cudaGetDeviceProperties
                    0.05%  659.38us       202  3.2640us     175ns  288.73us  cuDeviceGetAttribute
                    0.01%  108.91us         4  27.227us  16.445us  35.169us  cudaLaunchKernel
                    0.00%  58.968us         2  29.484us  22.121us  36.847us  cuDeviceGetName
                    0.00%  53.025us        14  3.7870us     713ns  17.133us  cudaSetDevice
                    0.00%  42.766us         2  21.383us  18.419us  24.347us  cudaStreamDestroy
                    0.00%  16.343us         2  8.1710us  2.6060us  13.737us  cuDeviceGetPCIBusId
                    0.00%  12.771us         2  6.3850us  5.3350us  7.4360us  cudaEventRecord
                    0.00%  6.9400us         2  3.4700us  3.3530us  3.5870us  cudaEventCreate
                    0.00%  5.7610us         2  2.8800us  2.8160us  2.9450us  cudaEventDestroy
                    0.00%  3.4180us         1  3.4180us  3.4180us  3.4180us  cudaGetDeviceCount
                    0.00%  1.7400us         4     435ns     329ns     608ns  cudaGetLastError
                    0.00%  1.6760us         3     558ns     331ns  1.0140us  cuDeviceGetCount
                    0.00%  1.4820us         4     370ns     172ns     712ns  cuDeviceGet
                    0.00%     892ns         2     446ns     433ns     459ns  cuDeviceTotalMem
                    0.00%     854ns         2     427ns     279ns     575ns  cuDeviceGetUuid
                    0.00%     397ns         1     397ns     397ns     397ns  cuModuleGetLoadingMode

==37768== Unified Memory profiling result:
Device "Tesla V100-SXM2-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
       8  2.0000MB  2.0000MB  2.0000MB  16.00000MB  1.684755ms  Host To Device
       4  2.0000MB  2.0000MB  2.0000MB  8.000000MB  678.7800us  Device To Host
Device "Tesla V100-SXM2-16GB (1)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
       8  2.0000MB  2.0000MB  2.0000MB  16.00000MB  1.635383ms  Host To Device
       4  2.0000MB  2.0000MB  2.0000MB  8.000000MB  769.3080us  Device To Host