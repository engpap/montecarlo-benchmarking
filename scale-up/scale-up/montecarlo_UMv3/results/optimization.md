# Starting point
./MonteCarlo Starting...

Using single CPU thread for multiple GPUs
MonteCarloMultiGPU
==================
Parallelization method  = streamed
Problem scaling         = weak
Number of GPUs          = 1
Total number of options = 1048576
Number of paths         = 262144
main(): generating input data...
main(): starting 1 host threads...
main(): GPU statistics, streamed
GPU Device #0: Tesla P100-SXM2-16GB
Options         : 1048576
Simulation paths: 262144

Total time (ms.): 2670.599121
        Note: This is elapsed time for all to compute.
Options per sec.: 392636.989849
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.810705E-04
Average reserve: 12.511987

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed

>>> Inside solverThread, initMonteCarloGPU took 7.50061 milliseconds
>>> Inside solverThread, MonteCarloGPU took 2670.6 milliseconds
./MonteCarlo Starting...

Using single CPU thread for multiple GPUs
MonteCarloMultiGPU
==================
Parallelization method  = streamed
Problem scaling         = weak
Number of GPUs          = 1
Total number of options = 1048576
Number of paths         = 262144
main(): generating input data...
main(): starting 1 host threads...
main(): GPU statistics, streamed
GPU Device #0: Tesla P100-SXM2-16GB
Options         : 1048576
Simulation paths: 262144

Total time (ms.): 2670.874023
        Note: This is elapsed time for all to compute.
Options per sec.: 392596.577300
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.810705E-04
Average reserve: 12.511987

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed

>>> Inside solverThread, initMonteCarloGPU took 7.55866 milliseconds
>>> Inside solverThread, MonteCarloGPU took 2670.87 milliseconds
./MonteCarlo Starting...

Using single CPU thread for multiple GPUs
==6797== NVPROF is profiling process 6797, command: ./MonteCarlo --method=streamed --scaling=weak
MonteCarloMultiGPU
==================
Parallelization method  = streamed
Problem scaling         = weak
Number of GPUs          = 1
Total number of options = 1048576
Number of paths         = 262144
main(): generating input data...
main(): starting 1 host threads...
main(): GPU statistics, streamed
GPU Device #0: Tesla P100-SXM2-16GB
Options         : 1048576
Simulation paths: 262144

Total time (ms.): 2670.811035
        Note: This is elapsed time for all to compute.
Options per sec.: 392605.836279
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.810705E-04
Average reserve: 12.511987

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed

>>> Inside solverThread, initMonteCarloGPU took 27.8676 milliseconds
>>> Inside solverThread, MonteCarloGPU took 2670.81 milliseconds
==6797== Profiling application: ./MonteCarlo --method=streamed --scaling=weak
==6797== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.74%  2.65797s         1  2.65797s  2.65797s  2.65797s  MonteCarloOneBlockPerOption(curandStateXORWOW*, __TOptionData const *, __TOptionValue*, int, int)
                    0.26%  6.8868ms         1  6.8868ms  6.8868ms  6.8868ms  rngSetupStates(curandStateXORWOW*, int)
      API calls:   92.90%  2.65809s         1  2.65809s  2.65809s  2.65809s  cudaEventSynchronize
                    5.65%  161.59ms         1  161.59ms  161.59ms  161.59ms  cudaStreamCreate
                    0.71%  20.373ms         3  6.7912ms  22.464us  20.326ms  cudaMallocManaged
                    0.37%  10.521ms         5  2.1042ms  57.703us  5.2222ms  cudaMemPrefetchAsync
                    0.24%  6.8880ms         1  6.8880ms  6.8880ms  6.8880ms  cudaDeviceSynchronize
                    0.10%  2.7719ms         3  923.97us  688.68us  1.0954ms  cudaFree
                    0.02%  647.85us         4  161.96us  108.62us  202.12us  cudaGetDeviceProperties
                    0.01%  160.12us       101  1.5850us     162ns  72.449us  cuDeviceGetAttribute
                    0.00%  70.240us         2  35.120us  27.817us  42.423us  cudaLaunchKernel
                    0.00%  31.421us         1  31.421us  31.421us  31.421us  cuDeviceGetName
                    0.00%  28.961us         7  4.1370us     647ns  13.066us  cudaSetDevice
                    0.00%  18.934us         1  18.934us  18.934us  18.934us  cudaStreamDestroy
                    0.00%  13.029us         1  13.029us  13.029us  13.029us  cuDeviceGetPCIBusId
                    0.00%  7.3680us         1  7.3680us  7.3680us  7.3680us  cudaEventCreate
                    0.00%  6.3190us         1  6.3190us  6.3190us  6.3190us  cudaEventRecord
                    0.00%  2.7180us         1  2.7180us  2.7180us  2.7180us  cudaEventDestroy
                    0.00%  2.5350us         3     845ns     285ns  1.9150us  cuDeviceGetCount
                    0.00%  1.1310us         1  1.1310us  1.1310us  1.1310us  cudaGetDeviceCount
                    0.00%     907ns         2     453ns     170ns     737ns  cuDeviceGet
                    0.00%     637ns         1     637ns     637ns     637ns  cuDeviceTotalMem
                    0.00%     551ns         2     275ns     250ns     301ns  cudaGetLastError
                    0.00%     387ns         1     387ns     387ns     387ns  cuModuleGetLoadingMode
                    0.00%     301ns         1     301ns     301ns     301ns  cuDeviceGetUuid

==6797== Unified Memory profiling result:
Device "Tesla P100-SXM2-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
       8  2.0000MB  2.0000MB  2.0000MB  16.00000MB  1.404932ms  Host To Device
       4  2.0000MB  2.0000MB  2.0000MB  8.000000MB  672.1440us  Device To Host
--------------------------------------------------------------------------------------------------------------------

# The first step

Applying the cudaMemAdviseSetAccessedBy hint before the prefetch of initMonteCarloGPU:

```c++
extern "C" void initMonteCarloGPU(TOptionPlan *plan)
{
    ...

    cudaMemAdvise(plan->rngStates, plan->gridSize * THREAD_N * sizeof(curandState), cudaMemAdviseSetAccessedBy, plan->device);
    // Prefetch rngStates the the device
    checkCudaErrors(cudaMemPrefetchAsync(plan->rngStates, plan->gridSize * THREAD_N * sizeof(curandState), plan->device));

    // place each device pathN random numbers apart on the random number sequence
    rngSetupStates<<<plan->gridSize, THREAD_N>>>(plan->rngStates, plan->device);

    ...
}
```

We obtained:

--------------------------------------------------------------------------------------------------------------------
./MonteCarlo Starting...

Using single CPU thread for multiple GPUs
MonteCarloMultiGPU
==================
Parallelization method  = streamed
Problem scaling         = weak
Number of GPUs          = 1
Total number of options = 1048576
Number of paths         = 262144
main(): generating input data...
main(): starting 1 host threads...
main(): GPU statistics, streamed
GPU Device #0: Tesla P100-SXM2-16GB
Options         : 1048576
Simulation paths: 262144

Total time (ms.): 2670.672119
        Note: This is elapsed time for all to compute.
Options per sec.: 392626.257819
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.810705E-04
Average reserve: 12.511987

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed

>>> Inside solverThread, initMonteCarloGPU took 7.52502 milliseconds
>>> Inside solverThread, MonteCarloGPU took 2670.67 milliseconds
./MonteCarlo Starting...

Using single CPU thread for multiple GPUs
==6392== NVPROF is profiling process 6392, command: ./MonteCarlo --method=streamed --scaling=weak
MonteCarloMultiGPU
==================
Parallelization method  = streamed
Problem scaling         = weak
Number of GPUs          = 1
Total number of options = 1048576
Number of paths         = 262144
main(): generating input data...
main(): starting 1 host threads...
main(): GPU statistics, streamed
GPU Device #0: Tesla P100-SXM2-16GB
Options         : 1048576
Simulation paths: 262144

Total time (ms.): 2670.544922
        Note: This is elapsed time for all to compute.
Options per sec.: 392644.958492
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.810705E-04
Average reserve: 12.511987

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed

>>> Inside solverThread, initMonteCarloGPU took 28.2437 milliseconds
>>> Inside solverThread, MonteCarloGPU took 2670.54 milliseconds
==6392== Profiling application: ./MonteCarlo --method=streamed --scaling=weak
==6392== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.74%  2.65792s         1  2.65792s  2.65792s  2.65792s  MonteCarloOneBlockPerOption(curandStateXORWOW*, __TOptionData const *, __TOptionValue*, int, int)
                    0.26%  6.9036ms         1  6.9036ms  6.9036ms  6.9036ms  rngSetupStates(curandStateXORWOW*, int)
      API calls:   92.94%  2.65805s         1  2.65805s  2.65805s  2.65805s  cudaEventSynchronize
                    5.60%  160.16ms         1  160.16ms  160.16ms  160.16ms  cudaStreamCreate
                    0.72%  20.726ms         3  6.9085ms  31.011us  20.645ms  cudaMallocManaged
                    0.36%  10.396ms         5  2.0793ms  56.865us  5.1483ms  cudaMemPrefetchAsync
                    0.24%  6.9040ms         1  6.9040ms  6.9040ms  6.9040ms  cudaDeviceSynchronize
                    0.10%  2.7594ms         3  919.81us  661.01us  1.1061ms  cudaFree
                    0.02%  541.95us         4  135.49us  108.42us  152.59us  cudaGetDeviceProperties
                    0.01%  155.81us       101  1.5420us     166ns  71.653us  cuDeviceGetAttribute
                    0.00%  76.978us         2  38.489us  30.171us  46.807us  cudaLaunchKernel
                    0.00%  30.639us         1  30.639us  30.639us  30.639us  cuDeviceGetName
                    0.00%  30.588us         7  4.3690us     575ns  15.589us  cudaSetDevice
                    0.00%  18.567us         1  18.567us  18.567us  18.567us  cudaStreamDestroy
                    0.00%  13.283us         1  13.283us  13.283us  13.283us  cuDeviceGetPCIBusId
                    0.00%  12.919us         1  12.919us  12.919us  12.919us  cudaMemAdvise
                    0.00%  6.4910us         1  6.4910us  6.4910us  6.4910us  cudaEventRecord
                    0.00%  3.0260us         1  3.0260us  3.0260us  3.0260us  cudaEventCreate
                    0.00%  2.6570us         3     885ns     325ns  1.9860us  cuDeviceGetCount
                    0.00%  2.2380us         1  2.2380us  2.2380us  2.2380us  cudaEventDestroy
                    0.00%     873ns         1     873ns     873ns     873ns  cudaGetDeviceCount
                    0.00%     793ns         2     396ns     327ns     466ns  cudaGetLastError
                    0.00%     715ns         2     357ns     167ns     548ns  cuDeviceGet
                    0.00%     343ns         1     343ns     343ns     343ns  cuDeviceTotalMem
                    0.00%     335ns         1     335ns     335ns     335ns  cuModuleGetLoadingMode
                    0.00%     285ns         1     285ns     285ns     285ns  cuDeviceGetUuid

==6392== Unified Memory profiling result:
Device "Tesla P100-SXM2-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
       8  2.0000MB  2.0000MB  2.0000MB  16.00000MB  1.414578ms  Host To Device
       4  2.0000MB  2.0000MB  2.0000MB  8.000000MB  650.6470us  Device To Host
--------------------------------------------------------------------------------------------------------------------

While setting cudaMemAdviseSetAccessedBy allowed us to obtain an average init time 0.06 ms worse. To get this time, we averaged 8 init times.


# The second step

./MonteCarlo Starting...

Using single CPU thread for multiple GPUs
MonteCarloMultiGPU
==================
Parallelization method  = streamed
Problem scaling         = weak
Number of GPUs          = 1
Total number of options = 1048576
Number of paths         = 262144
main(): generating input data...
main(): starting 1 host threads...
main(): GPU statistics, streamed
GPU Device #0: Tesla P100-SXM2-16GB
Options         : 1048576
Simulation paths: 262144

Total time (ms.): 2672.073975
        Note: This is elapsed time for all to compute.
Options per sec.: 392420.273527
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.810705E-04
Average reserve: 12.511987

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed

>>> Inside solverThread, initMonteCarloGPU took 7.54598 milliseconds
>>> Inside solverThread, MonteCarloGPU took 2672.07 milliseconds
./MonteCarlo Starting...

Using single CPU thread for multiple GPUs
==34561== NVPROF is profiling process 34561, command: ./MonteCarlo --method=streamed --scaling=weak
MonteCarloMultiGPU
==================
Parallelization method  = streamed
Problem scaling         = weak
Number of GPUs          = 1
Total number of options = 1048576
Number of paths         = 262144
main(): generating input data...
main(): starting 1 host threads...
main(): GPU statistics, streamed
GPU Device #0: Tesla P100-SXM2-16GB
Options         : 1048576
Simulation paths: 262144

Total time (ms.): 2672.823975
        Note: This is elapsed time for all to compute.
Options per sec.: 392310.159577
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.810705E-04
Average reserve: 12.511987

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed

>>> Inside solverThread, initMonteCarloGPU took 28.0916 milliseconds
>>> Inside solverThread, MonteCarloGPU took 2672.82 milliseconds
==34561== Profiling application: ./MonteCarlo --method=streamed --scaling=weak
==34561== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.75%  2.65832s         1  2.65832s  2.65832s  2.65832s  MonteCarloOneBlockPerOption(curandStateXORWOW*, __TOptionData const *, __TOptionValue*, int, int)
                    0.25%  6.7705ms         1  6.7705ms  6.7705ms  6.7705ms  rngSetupStates(curandStateXORWOW*, int)
      API calls:   92.65%  2.65862s         1  2.65862s  2.65862s  2.65862s  cudaEventSynchronize
                    5.85%  167.79ms         1  167.79ms  167.79ms  167.79ms  cudaStreamCreate
                    0.72%  20.701ms         3  6.9003ms  24.619us  20.612ms  cudaMallocManaged
                    0.36%  10.199ms         5  2.0397ms  66.957us  5.2961ms  cudaMemPrefetchAsync
                    0.24%  6.8054ms         1  6.8054ms  6.8054ms  6.8054ms  cudaDeviceSynchronize
                    0.10%  2.7712ms         3  923.73us  662.81us  1.0903ms  cudaFree
                    0.06%  1.6432ms         5  328.65us  5.1120us  1.6026ms  cudaMemAdvise
                    0.02%  571.25us         4  142.81us  108.60us  170.84us  cudaGetDeviceProperties
                    0.01%  160.70us       101  1.5910us     173ns  72.288us  cuDeviceGetAttribute
                    0.00%  58.275us         2  29.137us  27.003us  31.272us  cudaLaunchKernel
                    0.00%  31.588us         1  31.588us  31.588us  31.588us  cuDeviceGetName
                    0.00%  25.481us         7  3.6400us  1.1930us  10.628us  cudaSetDevice
                    0.00%  19.247us         1  19.247us  19.247us  19.247us  cudaStreamDestroy
                    0.00%  13.269us         1  13.269us  13.269us  13.269us  cuDeviceGetPCIBusId
                    0.00%  5.5000us         1  5.5000us  5.5000us  5.5000us  cudaEventRecord
                    0.00%  3.1210us         1  3.1210us  3.1210us  3.1210us  cudaEventCreate
                    0.00%  2.2750us         1  2.2750us  2.2750us  2.2750us  cudaEventDestroy
                    0.00%  1.9330us         3     644ns     259ns  1.3840us  cuDeviceGetCount
                    0.00%  1.2990us         1  1.2990us  1.2990us  1.2990us  cudaGetDeviceCount
                    0.00%     898ns         2     449ns     174ns     724ns  cuDeviceGet
                    0.00%     661ns         2     330ns     268ns     393ns  cudaGetLastError
                    0.00%     488ns         1     488ns     488ns     488ns  cuDeviceTotalMem
                    0.00%     371ns         1     371ns     371ns     371ns  cuModuleGetLoadingMode
                    0.00%     308ns         1     308ns     308ns     308ns  cuDeviceGetUuid

==34561== Unified Memory profiling result:
Device "Tesla P100-SXM2-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
       8  2.0000MB  2.0000MB  2.0000MB  16.00000MB  1.416337ms  Host To Device
       4  2.0000MB  2.0000MB  2.0000MB  8.000000MB  660.6170us  Device To Host
       8  2.0000MB  2.0000MB  2.0000MB  16.00000MB           -  Remote mapping from device
Total remote mappings to CPU: 8