## From CUDA9 to CUDA11
For carrying out the porting from CUDA 9 to CUDA 11 on a 2 GPUs virtual machine, we created switched to AD2 and created a new boot volume. We changed the version of CUDA by simply changing the CUDA directory in the makefile, from: <br />
```CUDA_DIR = /usr/local/cuda-9.1/```<br />
To:<br />
```CUDA_DIR = /usr/local/cuda-11.7/``` <br />
We were able to run the program with CUDA11.


Running Montecarlo with CUDA11 (version 11.7) produces an error indicating that the linker is unable to find several libraries that are required by the program. Specifically, the libraries <em>mpich</em> and <em>mpl</em> cannot be found. In detail, we obtained the following error:
```
rm -f MonteCarlo
/usr/local/cuda-11.7//bin/nvcc -arch=sm_60  -O3  -I. -I/usr/local/cuda-11.7//include -I/home/ubuntu/NVIDIA_GPU_Computing_SDK//C/common/inc -I../../common/inc/ -I/home/ubuntu/NVIDIA_GPU_Computing_SDK//shared/inc -I/home/ubuntu/opt/miniconda2/pkgs/mpich2-1.4.1p1-0//include -I/home/lian599/include/ -L. -L/home/ubuntu/NVIDIA_GPU_Computing_SDK//C/lib -L/usr/local/cuda-11.7//lib64/ -L/home/ubuntu/NVIDIA_GPU_Computing_SDK//shared/lib -L/home/ubuntu/opt/miniconda2/pkgs/mpich2-1.4.1p1-0//lib -L/home/lian599/lib/ -L/usr/lib/ -L/usr/lib64  -lcuda -lmpich -lmpl  -lstdc++ -lm -I../common/inc -lcurand MonteCarloMultiGPU.cpp MonteCarlo_kernel.cu MonteCarlo_gold.cpp multithreading.cpp -o MonteCarlo
/usr/bin/ld: cannot find -lmpich
/usr/bin/ld: cannot find -lmpl
collect2: error: ld returned 1 exit status
make: *** [Makefile:39: MonteCarlo] Error 1
$Run montecarlo:/usr/bin/time -f '%e' ./run_1g_strong.sh
scale-up,MTC,strong,1,0.0
$Run montecarlo:/usr/bin/time -f '%e' ./run_1g_weak.sh
scale-up,MTC,weak,1,0.0
```

By installing mpich library through ```sudo apt install mpich``` and then running <em>run_scale_up.py</em>, we were able to fix the errors and obtained:
```
rm -f MonteCarlo
/usr/local/cuda-11.7//bin/nvcc -arch=sm_60  -O3  -I. -I/usr/local/cuda-11.7//include -I/home/ubuntu/NVIDIA_GPU_Computing_SDK//C/common/inc -I../../common/inc/ -I/home/ubuntu/NVIDIA_GPU_Computing_SDK//shared/inc -I/home/ubuntu/opt/miniconda2/pkgs/mpich2-1.4.1p1-0//include -I/home/lian599/include/ -L. -L/home/ubuntu/NVIDIA_GPU_Computing_SDK//C/lib -L/usr/local/cuda-11.7//lib64/ -L/home/ubuntu/NVIDIA_GPU_Computing_SDK//shared/lib -L/home/ubuntu/opt/miniconda2/pkgs/mpich2-1.4.1p1-0//lib -L/home/lian599/lib/ -L/usr/lib/ -L/usr/lib64  -lcuda -lmpich -lmpl  -lstdc++ -lm -I../common/inc -lcurand MonteCarloMultiGPU.cpp MonteCarlo_kernel.cu MonteCarlo_gold.cpp multithreading.cpp -o MonteCarlo
$Run montecarlo:/usr/bin/time -f '%e' ./run_1g_strong.sh
scale-up,MTC,strong,1,1.452
$Run montecarlo:/usr/bin/time -f '%e' ./run_2g_strong.sh
scale-up,MTC,strong,2,1.114
$Run montecarlo:/usr/bin/time -f '%e' ./run_1g_weak.sh
scale-up,MTC,weak,1,1.446
$Run montecarlo:/usr/bin/time -f '%e' ./run_2g_weak.sh
scale-up,MTC,weak,2,1.884
```


Additionally, we were able to get more information by running:
```
  $ cd scale-up/$app$/
  $ make
  $ chmod +x run.sh
  $ ./run.sh
```

Indeed, we obtained:
```
./MonteCarlo Starting...

Using single CPU thread for multiple GPUs
MapSMtoCores for SM 7.0 is undefined.  Default to use 128 Cores/SM
MapSMtoCores for SM 7.0 is undefined.  Default to use 128 Cores/SM
MonteCarloMultiGPU
==================
Parallelization method  = streamed
Problem scaling         = weak
Number of GPUs          = 2
Total number of options = 2097152
Number of paths         = 262144
main(): generating input data...
main(): starting 2 host threads...
main(): GPU statistics, streamed
GPU Device #0: Tesla V100-SXM2-16GB
Options         : 1048576
Simulation paths: 262144
GPU Device #1: Tesla V100-SXM2-16GB
Options         : 1048576
Simulation paths: 262144

Total time (ms.): 1072.712036
        Note: This is elapsed time for all to compute.
Options per sec.: 1954999.971437
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.811470E-04
Average reserve: 12.473080

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed
```

Secondly, by running:
```
./run_1g_strong.sh
```
We obtained:
```
./MonteCarlo Starting...

Using single CPU thread for multiple GPUs
MapSMtoCores for SM 7.0 is undefined.  Default to use 128 Cores/SM
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
GPU Device #0: Tesla V100-SXM2-16GB
Options         : 1048576
Simulation paths: 262144

Total time (ms.): 1061.562012
        Note: This is elapsed time for all to compute.
Options per sec.: 987767.071942
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.810658E-04
Average reserve: 12.329600

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
MapSMtoCores for SM 7.0 is undefined.  Default to use 128 Cores/SM
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
GPU Device #0: Tesla V100-SXM2-16GB
Options         : 1048576
Simulation paths: 262144

Total time (ms.): 1051.198975
        Note: This is elapsed time for all to compute.
Options per sec.: 997504.778189
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.810658E-04
Average reserve: 12.329600

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed
```

Eventually, we executed the scripts for 2 GPUs: ```run_2g_strong.sh``` and ```run_2g_weak.sh```
Thus, by running:
```
./run_2g_strong.sh
```
We obtained:
```
./MonteCarlo Starting...

Using single CPU thread for multiple GPUs
MapSMtoCores for SM 7.0 is undefined.  Default to use 128 Cores/SM
MapSMtoCores for SM 7.0 is undefined.  Default to use 128 Cores/SM
MonteCarloMultiGPU
==================
Parallelization method  = streamed
Problem scaling         = strong
Number of GPUs          = 2
Total number of options = 1048576
Number of paths         = 262144
main(): generating input data...
main(): starting 2 host threads...
main(): GPU statistics, streamed
GPU Device #0: Tesla V100-SXM2-16GB
Options         : 524288
Simulation paths: 262144
GPU Device #1: Tesla V100-SXM2-16GB
Options         : 524288
Simulation paths: 262144

Total time (ms.): 545.804016
        Note: This is elapsed time for all to compute.
Options per sec.: 1921158.454397
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.807236E-04
Average reserve: 12.704886

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed
```

Eventually, by running:
```
./run_2g_weak.sh
```
We obtained:
```
./MonteCarlo Starting...

Using single CPU thread for multiple GPUs
MapSMtoCores for SM 7.0 is undefined.  Default to use 128 Cores/SM
MapSMtoCores for SM 7.0 is undefined.  Default to use 128 Cores/SM
MonteCarloMultiGPU
==================
Parallelization method  = streamed
Problem scaling         = weak
Number of GPUs          = 2
Total number of options = 2097152
Number of paths         = 262144
main(): generating input data...
main(): starting 2 host threads...
main(): GPU statistics, streamed
GPU Device #0: Tesla V100-SXM2-16GB
Options         : 1048576
Simulation paths: 262144
GPU Device #1: Tesla V100-SXM2-16GB
Options         : 1048576
Simulation paths: 262144

Total time (ms.): 1061.215942
        Note: This is elapsed time for all to compute.
Options per sec.: 1976178.378258
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.811470E-04
Average reserve: 12.473080

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed
```

By looking at the output we noticed the following message: <br>
```
MapSMtoCores for SM 7.0 is undefined.  Default to use 128 Cores/SM
```
which is generated by the <em>helper_cuda.h/helper_cuda_drvapi.h</em> file when the current SM version is not found in the `nGpuArchCoresPerSM` array. <br>
After adding the following line: <br>
`{ 0x70, 64 }, // Volta Generation (SM 7.0) V100 class`, <br>
taken from the updated [cuda-samples GitHub repository](https://github.com/NVIDIA/cuda-samples.git), we successfully removed the message from the output:
```

```


