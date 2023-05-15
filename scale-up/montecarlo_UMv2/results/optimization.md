# WORKING WITH HINTS

Unified Memory maintains a single active copy of any page, and migrations are triggered by user hints or page faults. However, sometimes it is useful to create multiple copies of the data or pin pages to system memory and enable zero-copy access. CUDA 8 provides the new `cudaMemAdvise()` API which provides a set of memory usage hints that allow finer grain control over managed allocations.
These hints are specified with an input parameter to `cudaMemAdvise()`, such as:
- cudaMemAdviseSetPreferredLocation
- cudaMemAdviseSetAccessedBy
- cudaMemAdviseSetReadMostly


---------------------------------------------------------------------------------------------------
The baseline is the code ported in Unified Memory but without any optimization.
The following table shows execution times of the baseline code, run with the streamed method, strong scaling, on 1 GPU.
---------------------------------------------------------------------------------------------------
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.68%  2.66031s         1  2.66031s  2.66031s  2.66031s  MonteCarloOneBlockPerOption(curandStateXORWOW*, __TOptionData const *, __TOptionValue*, int, int)
                    0.32%  8.5086ms         1  8.5086ms  8.5086ms  8.5086ms  rngSetupStates(curandStateXORWOW*, int)
      API calls:   93.19%  2.66032s         1  2.66032s  2.66032s  2.66032s  cudaEventSynchronize
                    5.65%  161.40ms         1  161.40ms  161.40ms  161.40ms  cudaStreamCreate
                    0.72%  20.658ms         3  6.8859ms  30.088us  20.574ms  cudaMallocManaged
                    0.30%  8.5092ms         1  8.5092ms  8.5092ms  8.5092ms  cudaDeviceSynchronize
                    0.10%  2.9480ms         3  982.66us  706.58us  1.1502ms  cudaFree
                    0.02%  578.54us         4  144.63us  120.88us  154.90us  cudaGetDeviceProperties
                    0.01%  154.85us       101  1.5330us     165ns  70.403us  cuDeviceGetAttribute
                    0.00%  84.588us         2  42.294us  40.821us  43.767us  cudaLaunchKernel
                    0.00%  28.766us         1  28.766us  28.766us  28.766us  cuDeviceGetName
                    0.00%  26.267us         7  3.7520us     739ns  10.426us  cudaSetDevice
                    0.00%  25.002us         1  25.002us  25.002us  25.002us  cuDeviceGetPCIBusId
                    0.00%  19.028us         1  19.028us  19.028us  19.028us  cudaStreamDestroy
                    0.00%  6.9380us         1  6.9380us  6.9380us  6.9380us  cudaEventRecord
                    0.00%  3.1460us         1  3.1460us  3.1460us  3.1460us  cudaEventCreate
                    0.00%  2.7570us         1  2.7570us  2.7570us  2.7570us  cudaEventDestroy
                    0.00%  1.8790us         3     626ns     227ns  1.4040us  cuDeviceGetCount
                    0.00%  1.0480us         1  1.0480us  1.0480us  1.0480us  cudaGetDeviceCount
                    0.00%     893ns         2     446ns     193ns     700ns  cuDeviceGet
                    0.00%     825ns         2     412ns     333ns     492ns  cudaGetLastError
                    0.00%     662ns         1     662ns     662ns     662ns  cuModuleGetLoadingMode
                    0.00%     437ns         1     437ns     437ns     437ns  cuDeviceTotalMem
                    0.00%     267ns         1     267ns     267ns     267ns  cuDeviceGetUuid

==26325== Unified Memory profiling result:
Device "Tesla P100-SXM2-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     141  116.20KB  4.0000KB  0.9922MB  16.00000MB  1.635900ms  Host To Device
      48  170.67KB  4.0000KB  0.9961MB  8.000000MB  699.1790us  Device To Host
     145         -         -         -           -  12.24735ms  Gpu page fault groups
Total CPU Page faults: 72
---------------------------------------------------------------------------------------------------
streamed,strong,1,1048576,262144, init: 8.74457, exec: 2671.78
---------------------------------------------------------------------------------------------------

## Getting improvements using cudaMemAdviseSetPreferredLocation hints

# The first step

By looking at the code, plan->rngStates is mostly used by the GPU, so we can use cudaMemAdviseSetPreferredLocation to tell the system that this data is mostly used by the devices. Hence, initMonteCarloGPU is modified as follows:

```c++
extern "C" void initMonteCarloGPU(TOptionPlan *plan)
{
    ...
    checkCudaErrors(cudaMallocManaged((void **)&plan->rngStates,
                                      plan->gridSize * THREAD_N * sizeof(curandState)));

    cudaMemAdvise(plan->rngStates, plan->gridSize * THREAD_N * sizeof(curandState), cudaMemAdviseSetPreferredLocation, plan->device);

    // place each device pathN random numbers apart on the random number sequence
    rngSetupStates<<<plan->gridSize, THREAD_N>>>(plan->rngStates, plan->device);    
    ...
}
```

This led an improvement of rngSetupStates kernel execution time. From an avarage of 8.5086ms to an average of 7.9416ms.
Here is the output of nvprof:
 0.30%  7.9416ms         1  7.9416ms  7.9416ms  7.9416ms  rngSetupStates(curandStateXORWOW*, int)

Observation: If we instead used the cudaMemAdviseSetAccessedBy hint, time would not have been improved. TODO: motivate this!!!!!!!!!!!!!!!!!

# The second step

The next step is to give some hints on the data used by the kernel MonteCarloOneBlockPerOption. This data is um_OptionData and um_CallValue.
They are used by both the device and the host. To achieve the highest performance, we tried different combinations of cudaMemAdviseSetPreferredLocation, cudaMemAdviseSetAccessedBy and cudaMemAdviseSetReadMostly hints. The best results were obtained by using a combination of cudaMemAdviseSetPreferredLocation and cudaMemAdviseSetReadMostly. The modfied code is as follows:


```c++
extern "C" void MonteCarloGPU(TOptionPlan *plan, cudaStream_t stream)
{   
    ...

    checkCudaErrors(cudaMemAdvise(plan->um_OptionData, sizeof(__TOptionData) * (plan->optionCount), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId)); 

    __TOptionData *optionData = (__TOptionData *)plan->um_OptionData;

    for (int i = 0; i < plan->optionCount; i++)
    {
        ...
    }

    checkCudaErrors(cudaMemAdvise(plan->um_OptionData, sizeof(__TOptionData) * (plan->optionCount), cudaMemAdviseUnsetPreferredLocation , cudaCpuDeviceId)); 

    
    checkCudaErrors(cudaMemAdvise(plan->um_OptionData, sizeof(__TOptionData) * (plan->optionCount), cudaMemAdviseSetPreferredLocation, plan->device)); 
    checkCudaErrors(cudaMemAdvise(plan->um_CallValue, sizeof(__TOptionValue) * (plan->optionCount), cudaMemAdviseSetPreferredLocation, plan->device));
    
    MonteCarloOneBlockPerOption<<<plan->gridSize, THREAD_N, 0, stream>>>(
        plan->rngStates,
        (__TOptionData *)(plan->um_OptionData),
        (__TOptionValue *)(plan->um_CallValue),
        plan->pathN,
        plan->optionCount);
    ...
}
```


```c++
extern "C" void closeMonteCarloGPU(TOptionPlan *plan)
{
    
    checkCudaErrors(cudaMemAdvise(plan->um_OptionData, sizeof(__TOptionData) * (plan->optionCount), cudaMemAdviseUnsetPreferredLocation , plan->device)); 
    checkCudaErrors(cudaMemAdvise(plan->um_CallValue, sizeof(__TOptionValue) * (plan->optionCount), cudaMemAdviseUnsetPreferredLocation , plan->device)); 

    checkCudaErrors(cudaMemAdvise(plan->um_CallValue, sizeof(__TOptionValue) * (plan->optionCount), cudaMemAdviseSetReadMostly, cudaCpuDeviceId)); 
    
    for (int i = 0; i < plan->optionCount; i++)
    {
        ...
    }
    ...
}
```

This led to a small improvement of MonteCarloOneBlockPerOption kernel execution time. From an average of 2.66031s to an average of 2.65986s.
Here is the output of nvprof:
---------------------------------------------------------------------------------------------------
>>> Inside solverThread, initMonteCarloGPU took 28.8067 milliseconds
>>> Inside solverThread, MonteCarloGPU took 2672 milliseconds
==34339== Profiling application: ./MonteCarlo --method=streamed --scaling=strong
==34339== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.70%  2.65986s         1  2.65986s  2.65986s  2.65986s  MonteCarloOneBlockPerOption(curandStateXORWOW*, __TOptionData const *, __TOptionValue*, int, int)
                    0.30%  7.9083ms         1  7.9083ms  7.9083ms  7.9083ms  rngSetupStates(curandStateXORWOW*, int)
      API calls:   93.06%  2.65987s         1  2.65987s  2.65987s  2.65987s  cudaEventSynchronize
                    5.80%  165.71ms         1  165.71ms  165.71ms  165.71ms  cudaStreamCreate
                    0.72%  20.668ms         3  6.8895ms  31.132us  20.557ms  cudaMallocManaged
                    0.28%  7.8989ms         1  7.8989ms  7.8989ms  7.8989ms  cudaDeviceSynchronize
                    0.10%  2.8911ms         3  963.71us  741.55us  1.1084ms  cudaFree
                    0.02%  645.22us         4  161.30us  112.37us  230.75us  cudaGetDeviceProperties
                    0.01%  198.86us         8  24.856us     791ns  123.80us  cudaMemAdvise
                    0.01%  158.93us       101  1.5730us     163ns  74.078us  cuDeviceGetAttribute
                    0.00%  102.03us         2  51.015us  49.736us  52.294us  cudaLaunchKernel
                    0.00%  38.055us         7  5.4360us     676ns  22.133us  cudaSetDevice
                    0.00%  32.790us         1  32.790us  32.790us  32.790us  cuDeviceGetName
                    0.00%  20.342us         1  20.342us  20.342us  20.342us  cudaStreamDestroy
                    0.00%  14.106us         1  14.106us  14.106us  14.106us  cuDeviceGetPCIBusId
                    0.00%  7.3400us         1  7.3400us  7.3400us  7.3400us  cudaEventRecord
                    0.00%  3.6570us         1  3.6570us  3.6570us  3.6570us  cudaEventCreate
                    0.00%  2.6730us         3     891ns     268ns  2.1140us  cuDeviceGetCount
                    0.00%  2.5740us         1  2.5740us  2.5740us  2.5740us  cudaEventDestroy
                    0.00%  1.1550us         1  1.1550us  1.1550us  1.1550us  cudaGetDeviceCount
                    0.00%     983ns         2     491ns     454ns     529ns  cudaGetLastError
                    0.00%     881ns         2     440ns     160ns     721ns  cuDeviceGet
                    0.00%     369ns         1     369ns     369ns     369ns  cuDeviceTotalMem
                    0.00%     320ns         1     320ns     320ns     320ns  cuModuleGetLoadingMode
                    0.00%     286ns         1     286ns     286ns     286ns  cuDeviceGetUuid

==34339== Unified Memory profiling result:
Device "Tesla P100-SXM2-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     141  116.20KB  4.0000KB  0.9922MB  16.00000MB  1.642841ms  Host To Device
      48  170.67KB  4.0000KB  0.9961MB  8.000000MB  716.1410us  Device To Host
      82         -         -         -           -  9.019436ms  Gpu page fault groups
Total CPU Page faults: 32
---------------------------------------------------------------------------------------------------


# Conclusion
Overall, the observable improvement was only obtained on hints applied to rngStates, whereas hints applied and showed in "The second step" paragraph, didn't show any great improvement.



# The second step revised
To better understand how the cudaMemAdvise improved the performance for each data, we tried to first apply changes to only um_optionData and then only to um_callValue.





!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// IL SEGUENTE SONO DATI PER I DEVELOPERS PER CAPIRE COSA NON HA FUNZIONATO. IN REALTA HO FATTO MOLTE ALTRE PROVE CHE NON HO MESSO QUI PERCHE' NON PORTAVANO MIGLIORAMENTI SIGNIFICATIVI. NON SO SE INSERIRLI NEL REPORT FINALE, OCCUPEREBBERO TROPPO SPAZIO.


## What worsened times

# cudaMemAdviseSetAccessedBy

By default, any CPU access of cudaMallocManaged allocations resident in GPU memory will trigger page faults and data migration. Applications can use cudaMemAdviseSetAccessedBy performance hint with cudaCpuDeviceId to enable direct access of GPU memory on supported systems.

By looking at the code um_CallValue is written by the GPU, for each option, and accessed by CPU when closing the MonteCarlo algorithm to recover the resulss. For this reason, wo we tried to insert that hint on plan->um_CallValue using cudaMemAdvise.

extern "C" void initMonteCarloGPU(TOptionPlan *plan)
{
    checkCudaErrors(cudaMallocManaged(&plan->um_OptionData, sizeof(__TOptionData) * (plan->optionCount)));
    checkCudaErrors(cudaMallocManaged(&plan->um_CallValue, sizeof(__TOptionValue) * (plan->optionCount)));

    // Applications can use cudaMemAdviseSetAccessedBy performance hint with cudaCpuDeviceId to enable direct access of GPU memory on supported systems.
    checkCudaErrors(cudaMemAdvise(plan->um_CallValue, sizeof(__TOptionValue) * (plan->optionCount), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId)); 
    
    ....
}

But we did not neither obtained performance improvements nor page faults reduction.

TODO: insert data, screenshots or numbers!!!

# cudaMemAdviseSetAccessedBy

The `cudaMemAdviseSetReadMostly` parameter allows to automatically duplicate data on a specified processor. Writing to such memory is allowed and doing so will invalidate all the copies, so it’s a very expensive operation. This is why it’s called read-mostly which tells the CUDA driver that such memory region will be mostly read from and only occasionally written to.

By looking at the code um_OptionData is read - but not written - by the GPU, for each option, and accessed by GPU when closing the MonteCarlo algorithm to recover the resulss. For this reason, wo we tried to insert that hint on plan->um_OptionData using cudaMemAdvise.


extern "C" void MonteCarloGPU(TOptionPlan *plan, cudaStream_t stream)
{
    ....

    checkCudaErrors(cudaMemAdvise(plan->um_OptionData, sizeof(__TOptionData) * (plan->optionCount), cudaMemAdviseSetReadMostly, plan->device)); 

    MonteCarloOneBlockPerOption<<<plan->gridSize, THREAD_N, 0, stream>>>(...);

    ...
}


But doing so, we insreased by 2 page faults the total gpu page fault groups:

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.68%  2.66038s         1  2.66038s  2.66038s  2.66038s  MonteCarloOneBlockPerOption(curandStateXORWOW*, __TOptionData const *, __TOptionValue*, int, int)
                    0.32%  8.5945ms         1  8.5945ms  8.5945ms  8.5945ms  rngSetupStates(curandStateXORWOW*, int)
      API calls:   93.08%  2.66039s         1  2.66039s  2.66039s  2.66039s  cudaEventSynchronize
                    5.68%  162.48ms         1  162.48ms  162.48ms  162.48ms  cudaStreamCreate
                    0.72%  20.554ms         3  6.8513ms  28.942us  20.474ms  cudaMallocManaged
                    0.30%  8.5954ms         1  8.5954ms  8.5954ms  8.5954ms  cudaDeviceSynchronize
                    0.12%  3.4663ms         3  1.1554ms  702.69us  1.6206ms  cudaFree
                    0.06%  1.6839ms         1  1.6839ms  1.6839ms  1.6839ms  cudaMemAdvise
                    0.02%  594.10us         4  148.52us  109.53us  188.24us  cudaGetDeviceProperties
                    0.01%  158.52us       101  1.5690us     167ns  73.647us  cuDeviceGetAttribute
                    0.00%  75.666us         2  37.833us  36.704us  38.962us  cudaLaunchKernel
                    0.00%  32.770us         1  32.770us  32.770us  32.770us  cuDeviceGetName
                    0.00%  28.478us         7  4.0680us     872ns  10.333us  cudaSetDevice
                    0.00%  21.549us         1  21.549us  21.549us  21.549us  cudaStreamDestroy
                    0.00%  13.113us         1  13.113us  13.113us  13.113us  cuDeviceGetPCIBusId
                    0.00%  6.4340us         1  6.4340us  6.4340us  6.4340us  cudaEventRecord
                    0.00%  3.8490us         1  3.8490us  3.8490us  3.8490us  cudaEventCreate
                    0.00%  2.9630us         1  2.9630us  2.9630us  2.9630us  cudaEventDestroy
                    0.00%  2.2840us         3     761ns     264ns  1.7260us  cuDeviceGetCount
                    0.00%  1.1090us         1  1.1090us  1.1090us  1.1090us  cudaGetDeviceCount
                    0.00%  1.0390us         2     519ns     183ns     856ns  cuDeviceGet
                    0.00%     923ns         2     461ns     393ns     530ns  cudaGetLastError
                    0.00%     465ns         1     465ns     465ns     465ns  cuDeviceTotalMem
                    0.00%     378ns         1     378ns     378ns     378ns  cuModuleGetLoadingMode
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetUuid

==24485== Unified Memory profiling result:
Device "Tesla P100-SXM2-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     141  116.20KB  4.0000KB  0.9922MB  16.00000MB  1.677275ms  Host To Device
      48  170.67KB  4.0000KB  0.9961MB  8.000000MB  720.0750us  Device To Host
     151         -         -         -           -  11.75170ms  Gpu page fault groups
Total CPU Page faults: 72
streamed,strong,1,1048576,262144,8.68292,2679.6



-------------------------------------------------------------------------------------------

By using the optimization on rngStates that we did in the firtt example of this post, plus using hints in MonteCarloGPU and closeMonteCarloGPU both with cudaMemAdviseSetAccessedBy and cudaMemAdviseUnsetPreferredLocation, we did not obtained performance improvement.

Here below the example using cudaMemAdviseSetAccessedBy hint for the MonteCarloGPU function and closeMonteCarloGPU function.
best on rngStates
plus
extern "C" void MonteCarloGPU(TOptionPlan *plan, cudaStream_t stream)
{   
    ...

    checkCudaErrors(cudaMemAdvise(plan->um_OptionData, sizeof(__TOptionData) * (plan->optionCount), cudaMemAdviseSetAccessedBy, plan->device)); 
    checkCudaErrors(cudaMemAdvise(plan->um_CallValue, sizeof(__TOptionValue) * (plan->optionCount), cudaMemAdviseSetAccessedBy, plan->device)); 
    
    MonteCarloOneBlockPerOption<<<plan->gridSize, THREAD_N, 0, stream>>>(
        plan->rngStates,
        (__TOptionData *)(plan->um_OptionData),
        (__TOptionValue *)(plan->um_CallValue),
        plan->pathN,
        plan->optionCount);
    ...
}


extern "C" void closeMonteCarloGPU(TOptionPlan *plan)
{
    checkCudaErrors(cudaMemAdvise(plan->um_OptionData, sizeof(__TOptionData) * (plan->optionCount), cudaMemAdviseUnsetAccessedBy, plan->device)); 
    checkCudaErrors(cudaMemAdvise(plan->um_CallValue, sizeof(__TOptionValue) * (plan->optionCount), cudaMemAdviseUnsetAccessedBy, plan->device)); 

    checkCudaErrors(cudaMemAdvise(plan->um_CallValue, sizeof(__TOptionValue) * (plan->optionCount), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId)); 

    ...

    checkCudaErrors(cudaFree(plan->rngStates));
    checkCudaErrors(cudaFree(plan->um_CallValue));
    checkCudaErrors(cudaFree(plan->um_OptionData));
}

We got:

>>> Inside solverThread, initMonteCarloGPU took 28.7818 milliseconds
>>> Inside solverThread, MonteCarloGPU took 2676.92 milliseconds
==31411== Profiling application: ./MonteCarlo --method=streamed --scaling=strong
==31411== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.71%  2.66123s         1  2.66123s  2.66123s  2.66123s  MonteCarloOneBlockPerOption(curandStateXORWOW*, __TOptionData const *, __TOptionValue*, int, int)
                    0.29%  7.8303ms         1  7.8303ms  7.8303ms  7.8303ms  rngSetupStates(curandStateXORWOW*, int)
      API calls:   92.96%  2.66124s         1  2.66124s  2.66124s  2.66124s  cudaEventSynchronize
                    5.84%  167.16ms         1  167.16ms  167.16ms  167.16ms  cudaStreamCreate
                    0.72%  20.702ms         3  6.9006ms  56.300us  20.580ms  cudaMallocManaged
                    0.27%  7.8325ms         1  7.8325ms  7.8325ms  7.8325ms  cudaDeviceSynchronize
                    0.12%  3.3258ms         3  1.1086ms  715.53us  1.5031ms  cudaFree
                    0.06%  1.7460ms         6  291.00us  3.1930us  1.6188ms  cudaMemAdvise
                    0.02%  568.84us         4  142.21us  108.82us  161.46us  cudaGetDeviceProperties
                    0.01%  160.21us       101  1.5860us     158ns  72.247us  cuDeviceGetAttribute
                    0.00%  76.375us         2  38.187us  36.490us  39.885us  cudaLaunchKernel
                    0.00%  30.441us         1  30.441us  30.441us  30.441us  cuDeviceGetName
                    0.00%  25.589us         7  3.6550us     653ns  10.631us  cudaSetDevice
                    0.00%  21.394us         1  21.394us  21.394us  21.394us  cudaStreamDestroy
                    0.00%  13.206us         1  13.206us  13.206us  13.206us  cuDeviceGetPCIBusId
                    0.00%  6.4610us         1  6.4610us  6.4610us  6.4610us  cudaEventRecord
                    0.00%  3.3480us         1  3.3480us  3.3480us  3.3480us  cudaEventCreate
                    0.00%  2.6180us         1  2.6180us  2.6180us  2.6180us  cudaEventDestroy
                    0.00%  2.1780us         3     726ns     276ns  1.6160us  cuDeviceGetCount
                    0.00%     913ns         1     913ns     913ns     913ns  cudaGetDeviceCount
                    0.00%     907ns         2     453ns     448ns     459ns  cudaGetLastError
                    0.00%     869ns         2     434ns     177ns     692ns  cuDeviceGet
                    0.00%     387ns         1     387ns     387ns     387ns  cuModuleGetLoadingMode
                    0.00%     379ns         1     379ns     379ns     379ns  cuDeviceTotalMem
                    0.00%     296ns         1     296ns     296ns     296ns  cuDeviceGetUuid

==31411== Unified Memory profiling result:
Device "Tesla P100-SXM2-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      48  170.67KB  4.0000KB  0.9961MB  8.000000MB  719.3730us  Device To Host
      48         -         -         -           -  3.304658ms  Gpu page fault groups
       8  2.0000MB  2.0000MB  2.0000MB  16.00000MB           -  Remote mapping from device
Total CPU Page faults: 72
Total remote mappings to CPU: 8

---

