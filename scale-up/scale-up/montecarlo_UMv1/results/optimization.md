## How we optimized the porting to Unified Memory

Optimization of page faulting is a technique used to improve the performance of Unified Memory in GPU computing. Unified Memory allows the GPU to access any page of the entire system memory and migrate data on-demand to its own memory for high bandwidth access. However, this process can introduce overhead due to the complex page fault handling mechanism. By minimizing page faults during CUDA kernel execution and providing enough information about the program’s access pattern to the driver, it is possible to improve the performance of Unified Memory.

There are different approaches to migrating the data from the system memory to the GPU memory. For example:
   1. On-demand migration by passing the cudaMallocManaged pointer directly to the kernel;
   2. Prefetching the data before the kernel launch by calling cudaMemPrefetchAsync on the cudaMallocManaged pointer;
   3. Copying the data from cudaMallocHost to a preallocated cudaMalloc buffer on the GPU using cudaMemcpyAsync.

In this version of Unified Memory we focused on point 2.

# Prefetching the data
The cudaMemPrefetchAsync function is used to prefetch data before launching a kernel. This function takes a pointer to the data allocated with cudaMallocManaged and transfers it to the GPU memory. By prefetching the data before launching the kernel, the data is already available in the GPU memory when the kernel starts executing, which can improve performance.

To achieve this we modified the initMonteCarloGPU function by inserting the prefetch of rngStates to the device (GPU).
<br/>Inserted lines are 142-144:<br/>
![alt text](https://github.com/engpap/montecarlo-benchmarking/blob/porting-um/scale-up/scale-up/report/assets/montecarlo_UMv1/initMonteCarloGPU.png?raw=true)


Additionally, we modified the MonteCarloGPU function by inserting the prefetch of the option data and the call value to the device (GPU).
<br/>Inserted lines are 199-202:<br/>
![alt text](https://github.com/engpap/montecarlo-benchmarking/blob/porting-um/scale-up/scale-up/report/assets/montecarlo_UMv1/MonteCarloGPU.png?raw=true)


After doing these steps, we obtained the following output running <em>run_scale_up.py</em>: <br>
```
rm -f MonteCarlo
/usr/local/cuda-11.7//bin/nvcc -arch=sm_60  -O3  -I. -I/usr/local/cuda-11.7//include -I/home/ubuntu/NVIDIA_GPU_Computing_SDK//C/common/inc -I../../common/inc/ -I/home/ubuntu/NVIDIA_GPU_Computing_SDK//shared/inc -I/home/ubuntu/opt/miniconda2/pkgs/mpich2-1.4.1p1-0//include -I/home/lian599/include/ -L. -L/home/ubuntu/NVIDIA_GPU_Computing_SDK//C/lib -L/usr/local/cuda-11.7//lib64/ -L/home/ubuntu/NVIDIA_GPU_Computing_SDK//shared/lib -L/home/ubuntu/opt/miniconda2/pkgs/mpich2-1.4.1p1-0//lib -L/home/lian599/lib/ -L/usr/lib/ -L/usr/lib64  -lcuda -lmpich -lmpl  -lstdc++ -lm -I../common/inc -lcurand MonteCarloMultiGPU.cpp MonteCarlo_kernel.cu MonteCarlo_gold.cpp multithreading.cpp -o MonteCarlo

$Run montecarlo_UMv1:/usr/bin/time -f '%e' ./run_1g_strong.sh
scale-up,MTC,strong,1,5154.000000000001

$Run montecarlo_UMv1:/usr/bin/time -f '%e' ./run_1g_weak.sh
scale-up,MTC,weak,1,5156.000000000001
```

Then, by running 
```
nvprof ./run.sh
```
we observed the following output:

```
rm -f MonteCarlo
/usr/local/cuda-11.7//bin/nvcc -arch=sm_60  -O3  -I. -I/usr/local/cuda-11.7//include -I/home/ubuntu/NVIDIA_GPU_Computing_SDK//C/common/inc -I../../common/inc/ -I/home/ubuntu/NVIDIA_GPU_Computing_SDK//shared/inc -I/home/ubuntu/opt/miniconda2/pkgs/mpich2-1.4.1p1-0//include -I/home/lian599/include/ -L. -L/home/ubuntu/NVIDIA_GPU_Computing_SDK//C/lib -L/usr/local/cuda-11.7//lib64/ -L/home/ubuntu/NVIDIA_GPU_Computing_SDK//shared/lib -L/home/ubuntu/opt/miniconda2/pkgs/mpich2-1.4.1p1-0//lib -L/home/lian599/lib/ -L/usr/lib/ -L/usr/lib64  -lcuda -lmpich -lmpl  -lstdc++ -lm -I../common/inc -lcurand MonteCarloMultiGPU.cpp MonteCarlo_kernel.cu MonteCarlo_gold.cpp multithreading.cpp -o MonteCarlo
./MonteCarlo Starting...

Using single CPU thread for multiple GPUs
==40417== NVPROF is profiling process 40417, command: ./MonteCarlo
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

Total time (ms.): 2698.854980
        Note: This is elapsed time for all to compute.
Options per sec.: 388526.248201
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.810705E-04
Average reserve: 12.511987

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed
==40417== Profiling application: ./MonteCarlo
==40417== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.72%  2.68361s         1  2.68361s  2.68361s  2.68361s  MonteCarloOneBlockPerOption(curandStateXORWOW*, __TOptionData const *, __TOptionValue*, int, int)
                    0.28%  7.6108ms         1  7.6108ms  7.6108ms  7.6108ms  rngSetupStates(curandStateXORWOW*, int)
      API calls:   93.07%  2.68369s         1  2.68369s  2.68369s  2.68369s  cudaEventSynchronize
                    5.72%  165.07ms         1  165.07ms  165.07ms  165.07ms  cudaStreamCreate
                    0.72%  20.628ms         3  6.8760ms  29.956us  20.547ms  cudaMallocManaged
                    0.26%  7.6116ms         1  7.6116ms  7.6116ms  7.6116ms  cudaDeviceSynchronize
                    0.11%  3.0753ms         3  1.0251ms  862.71us  1.1281ms  cudaFree
                    0.08%  2.3383ms         3  779.44us  175.19us  1.7644ms  cudaMemPrefetchAsync
                    0.02%  587.70us         4  146.92us  127.09us  168.41us  cudaGetDeviceProperties
                    0.01%  245.62us       101  2.4310us     169ns  131.53us  cuDeviceGetAttribute
                    0.00%  66.242us         2  33.121us  30.218us  36.024us  cudaLaunchKernel
                    0.00%  26.710us         1  26.710us  26.710us  26.710us  cuDeviceGetName
                    0.00%  22.934us         7  3.2760us     767ns  7.4810us  cudaSetDevice
                    0.00%  21.022us         1  21.022us  21.022us  21.022us  cudaStreamDestroy
                    0.00%  6.2730us         1  6.2730us  6.2730us  6.2730us  cudaEventRecord
                    0.00%  3.0330us         1  3.0330us  3.0330us  3.0330us  cudaEventCreate
                    0.00%  2.9350us         1  2.9350us  2.9350us  2.9350us  cudaEventDestroy
                    0.00%  2.2350us         3     745ns     257ns  1.6970us  cuDeviceGetCount
                    0.00%  1.0430us         2     521ns     515ns     528ns  cudaGetLastError
                    0.00%  1.0030us         1  1.0030us  1.0030us  1.0030us  cudaGetDeviceCount
                    0.00%     797ns         2     398ns     178ns     619ns  cuDeviceGet
                    0.00%     556ns         1     556ns     556ns     556ns  cuDeviceGetUuid
                    0.00%     409ns         1     409ns     409ns     409ns  cuModuleGetLoadingMode
                    0.00%     392ns         1     392ns     392ns     392ns  cuDeviceTotalMem

==40417== Unified Memory profiling result:
Device "Tesla P100-SXM2-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
       8  2.0000MB  2.0000MB  2.0000MB  16.00000MB  1.428626ms  Host To Device
      48  170.67KB  4.0000KB  0.9961MB  8.000000MB  726.1870us  Device To Host
Total CPU Page faults: 72
======== Warning: No profile data collected.
```

# What is changed?
The main difference between the profiling outputs of the non-optimized version and this one is the number and size of the host to device memory transfers, with the first output having more transfers but smaller sizes, and the second output having fewer transfers but larger sizes. 
Furthermore, in this output we can observe zero GPU page fault groups compared to hundreds in the non-optimized output. This proves reduction of GPU page fault groups. <br>

# More outputs
By running:
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

Total time (ms.): 2899.045898
        Note: This is elapsed time for all to compute.
Options per sec.: 361696.929519
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.810705E-04
Average reserve: 12.511987

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed
```

While, by running:
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

Total time (ms.): 2907.273926
        Note: This is elapsed time for all to compute.
Options per sec.: 360673.272202
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.810705E-04
Average reserve: 12.511987

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed
```

--------------------------

# Prefetching data to device - Removing GPU page faults TODO: this is confirmed only for 1g, try other instances
By observing the profiling output, we noticed that for rngState there was a considerable amount of page faults, which caused a performance loss. So to remove those page faults, we introduced prefetching and eventually we were able to observe a better performance.
TODO: add screenshots
A similar reasoning was applied for the page faults caused by the transfer of the optionData variable and callValue variables. Even if there were only 2 page faults per GPU caused by the memory transfer of the callValue variable,  after analyzing the profiling output, we observed that the prefetching was still convenient: having those 2 page faults required more time than prefetching.
TODO: other screenshots
In conclusion, we gained some advantage making the init and execution time smaller.

## optimizations obtained
By appling these optimizations, we were able to reduce the number of *Unified Memory memcpy Host to Device* from 17 to 1, gaining almost 30% total time reduction for these memory transfers.

# Prefetching data to host - Removing CPU page faults  TODO: this is confirmed only for 1g, try other instances
In the *CPU Page Faults* section of the profiling output we noticed that these page faults occured on the MonteCarloGPU and on the closeMonteCarloGPU methods. Observing the code, we realized that the um_optionData and the um_callValue variables were being accessed by the CPU, respectively on MonteCarloGPU and closeMonteCarloGPU.

## streamed method
We applied prefetching, setting the destination device to the macro cudaCpuDeviceId (-1) and eventually we managed to remove completely all CPU page faults. 

## threaded method
Unfortunately during the multithreaded execution, some Write Page Faults remained even after prefetching the data.
We discovered that this was due to the way *cudaMemPrefetchAsync* is implemented, that is it doesn't interrupt any other host process while migrating the data. To overcome this problem, we added *cudaStreamSynchronize*, passing the default stream (the only one used in the multithreaded execution). By doing so, we eventually managed to remove all remaining page faults. However, this caused the time elapsed between the two kernel calls (rngSetupStates and MonteCarloOneBlockPerOption) to increase almost 4-fold.

## optimizations obtained
By appling these optimizations, we were able to reduce the number of *Unified Memory memcpy Device to Host* from 10 to 1, gaining 15% total time reduction for these memory transfers.
Furthermore, by looking at the profiling we noticed that while the loop at lines 199-210 was running, the CUDA API remained idle after completing the prefetch of optionData. We exploited this timeslot to anticipate the prefetching of callValue, that is therefore migrated while the host executes the loop. 




