## Porting to Unified Memory


After doing these steps, we obtained the following output runnin <em>run_scale_up.py</em>: <br>
```
rm -f MonteCarlo
/usr/local/cuda-11.7//bin/nvcc -arch=sm_60  -O3  -I. -I/usr/local/cuda-11.7//include -I/home/ubuntu/NVIDIA_GPU_Computing_SDK//C/common/inc -I../../common/inc/ -I/home/ubuntu/NVIDIA_GPU_Computing_SDK//shared/inc -I/home/ubuntu/opt/miniconda2/pkgs/mpich2-1.4.1p1-0//include -I/home/lian599/include/ -L. -L/home/ubuntu/NVIDIA_GPU_Computing_SDK//C/lib -L/usr/local/cuda-11.7//lib64/ -L/home/ubuntu/NVIDIA_GPU_Computing_SDK//shared/lib -L/home/ubuntu/opt/miniconda2/pkgs/mpich2-1.4.1p1-0//lib -L/home/lian599/lib/ -L/usr/lib/ -L/usr/lib64  -lcuda -lmpich -lmpl  -lstdc++ -lm -I../common/inc -lcurand MonteCarloMultiGPU.cpp MonteCarlo_kernel.cu MonteCarlo_gold.cpp multithreading.cpp -o MonteCarlo

$Run montecarlo_UM:/usr/bin/time -f '%e' ./run_1g_strong.sh
scale-up,MTC,strong,1,3051.9999999999995ms
$Run montecarlo_UM:/usr/bin/time -f '%e' ./run_1g_weak.sh
scale-up,MTC,weak,1,3046.0000000000005ms
```

Then, by running 
```
nvprof ./run.sh
```
we observed the following output:

```
./MonteCarlo Starting...

Using single CPU thread for multiple GPUs
==16079== NVPROF is profiling process 16079, command: ./MonteCarlo
MonteCarloMultiGPU

...

Test passed
==16079== Profiling application: ./MonteCarlo
==16079== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.65%  2.67992s         1  2.67992s  2.67992s  2.67992s  MonteCarloOneBlockPerOption(curandStateXORWOW*, __TOptionData const *, __TOptionValue*, int, int)
                    0.35%  9.4404ms         1  9.4404ms  9.4404ms  9.4404ms  rngSetupStates(curandStateXORWOW*, int)
      API calls:   92.76%  2.67992s         1  2.67992s  2.67992s  2.67992s  cudaEventSynchronize
                    6.06%  175.12ms         1  175.12ms  175.12ms  175.12ms  cudaStreamCreate
                    0.71%  20.566ms         3  6.8552ms  31.854us  20.469ms  cudaMallocManaged
                    0.33%  9.4336ms         1  9.4336ms  9.4336ms  9.4336ms  cudaDeviceSynchronize
                    0.11%  3.0940ms         3  1.0313ms  784.53us  1.1667ms  cudaFree
                    0.02%  566.34us         4  141.58us  110.69us  163.06us  cudaGetDeviceProperties
                    0.01%  161.38us       101  1.5970us     160ns  67.459us  cuDeviceGetAttribute
                    0.00%  90.814us         2  45.407us  44.511us  46.303us  cudaLaunchKernel
                    0.00%  37.987us         7  5.4260us     845ns  20.880us  cudaSetDevice
                    0.00%  25.524us         1  25.524us  25.524us  25.524us  cuDeviceGetName
                    0.00%  19.540us         1  19.540us  19.540us  19.540us  cudaStreamDestroy
                    0.00%  14.319us         1  14.319us  14.319us  14.319us  cudaEventRecord
                    0.00%  7.6700us         3  2.5560us     236ns  7.1180us  cuDeviceGetCount
                    0.00%  7.4360us         1  7.4360us  7.4360us  7.4360us  cudaEventCreate
                    0.00%  3.1270us         1  3.1270us  3.1270us  3.1270us  cudaEventDestroy
                    0.00%     892ns         1     892ns     892ns     892ns  cudaGetDeviceCount
                    0.00%     818ns         1     818ns     818ns     818ns  cuModuleGetLoadingMode
                    0.00%     801ns         2     400ns     230ns     571ns  cuDeviceGet
                    0.00%     723ns         2     361ns     300ns     423ns  cudaGetLastError
                    0.00%     515ns         1     515ns     515ns     515ns  cuDeviceTotalMem
                    0.00%     280ns         1     280ns     280ns     280ns  cuDeviceGetUuid

==16079== Unified Memory profiling result:
Device "Tesla P100-SXM2-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     141  116.20KB  4.0000KB  0.9922MB  16.00000MB  1.631895ms  Host To Device
      48  170.67KB  4.0000KB  0.9961MB  8.000000MB  689.3860us  Device To Host
     153         -         -         -           -  12.77144ms  Gpu page fault groups
Total CPU Page faults: 72
```
which effectively confirms that the program is running using Unified Memory. <br>

Secondly, by running:
```
./run_1g_strong.sh
```
We obtained:

```
./MonteCarlo Starting...

Using single CPU thread for multiple GPUs
MonteCarloMultiGPU
==================
Parallelization method  = streamed
Problem scaling         = strong
Number of GPUs          = 1
Total number of options = 1048576
Number of paths         = 262144
main(): generating input data...
main(): starting 1 host threads...
main(): GPU statistics, streamed
GPU Device #0: Tesla P100-SXM2-16GB
Options         : 1048576
Simulation paths: 262144

Total time (ms.): 2690.233887
        Note: This is elapsed time for all to compute.
Options per sec.: 389771.315117
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.810705E-04
Average reserve: 12.511987

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed
```

Thirdly, by running:
```
./run_1g_weak.sh
```
We obtained:
```
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

Total time (ms.): 2680.042969
        Note: This is elapsed time for all to compute.
Options per sec.: 391253.428481
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.810705E-04
Average reserve: 12.511987

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed
```