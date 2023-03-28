# MonteCarlo Benchmarking
This is a project of the [High Performance Processors and Systems](https://www4.ceda.polimi.it/manifesti/manifesti/controller/ManifestoPublic.do?EVN_DETTAGLIO_RIGA_MANIFESTO=evento&aa=2022&k_cf=225&k_corso_la=481&k_indir=T2A&codDescr=089185&lang=EN&semestre=2&idGruppo=4474&idRiga=281811) course of Politecnico Di Milano.

This repository contains the evaluation and implementation of MonteCarlo workload on multi-GPU systems via Unified Memory.
The project is based on the [Tartan benchmarking suite](https://github.com/uuudown/Tartan/blob/master/IISWC-18.pdf)

## From CUDA9 to CUDA11
Running the original Tartan code for Montecarlo produces an error indicating that the version specified in the <em>shared.mk</em> makefile is CUDA9 (version 9.1). In detail, we obtained the following error:
```
rm -f MonteCarlo
/usr/local/cuda-9.1//bin/nvcc -arch=sm_60  -O3  -I. -I/usr/local/cuda-9.1//include -I/home/ubuntu/NVIDIA_GPU_Computing_SDK//C/common/inc -I../../common/inc/ -I/home/ubuntu/NVIDIA_GPU_Computing_SDK//shared/inc -I/home/ubuntu/opt/miniconda2/pkgs/mpich2-1.4.1p1-0//include -I/home/lian599/include/ -L. -L/home/ubuntu/NVIDIA_GPU_Computing_SDK//C/lib -L/usr/local/cuda-9.1//lib64/ -L/home/ubuntu/NVIDIA_GPU_Computing_SDK//shared/lib -L/home/ubuntu/opt/miniconda2/pkgs/mpich2-1.4.1p1-0//lib -L/home/lian599/lib/ -L/usr/lib/ -L/usr/lib64  -lcuda -lmpich -lmpl  -lstdc++ -lm -I../common/inc -lcurand MonteCarloMultiGPU.cpp MonteCarlo_kernel.cu MonteCarlo_gold.cpp multithreading.cpp -o MonteCarlo
/bin/bash: /usr/local/cuda-9.1//bin/nvcc: No such file or directory
make: *** [Makefile:39: MonteCarlo] Error 127
$Run montecarlo:/usr/bin/time -f '%e' ./run_1g_strong.sh
scale-up,MTC,strong,1,0.0
$Run montecarlo:/usr/bin/time -f '%e' ./run_1g_weak.sh
scale-up,MTC,weak,1,0.0
```

By simply changing the CUDA directory in the makefile, from: <br />
```CUDA_DIR = /usr/local/cuda-9.1/```<br />
To:<br />
```CUDA_DIR = /usr/local/cuda/``` <br />
We were able to run the program with CUDA11.


Running Montecarlo with CUDA11 (version 11.7) produces an error indicating that the linker is unable to find several libraries that are required by the program. Specifically, the libraries <em>lcutil_x86_64</em>, <em>lmpich</em>, <em>lmpl</em>, and <em>lnccl</em> cannot be found. In detail, we obtained the following error:
```
rm -f MonteCarlo
/usr/local/cuda-11.7//bin/nvcc -arch=sm_60  -O3  -I. -I/usr/local/cuda-11.7//include -I/home/ubuntu/NVIDIA_GPU_Computing_SDK//C/common/inc -I../../common/inc/ -I/home/ubuntu/NVIDIA_GPU_Computing_SDK//shared/inc -I/home/ubuntu/opt/miniconda2/pkgs/mpich2-1.4.1p1-0//include -I/home/lian599/include/ -L. -L/home/ubuntu/NVIDIA_GPU_Computing_SDK//C/lib -L/usr/local/cuda-11.7//lib64/ -L/home/ubuntu/NVIDIA_GPU_Computing_SDK//shared/lib -L/home/ubuntu/opt/miniconda2/pkgs/mpich2-1.4.1p1-0//lib -L/home/lian599/lib/ -L/usr/lib/ -L/usr/lib64  -lcutil_x86_64 -lcuda -lmpich -lmpl -lnccl -lstdc++ -lm -I../common/inc -lcurand MonteCarloMultiGPU.cpp MonteCarlo_kernel.cu MonteCarlo_gold.cpp multithreading.cpp -o MonteCarlo
/usr/bin/ld: cannot find -lcutil_x86_64
/usr/bin/ld: cannot find -lmpich
/usr/bin/ld: cannot find -lmpl
/usr/bin/ld: cannot find -lnccl
collect2: error: ld returned 1 exit status
make: *** [Makefile:39: MonteCarlo] Error 1
$Run montecarlo:/usr/bin/time -f '%e' ./run_1g_strong.sh
scale-up,MTC,strong,1,0.0
$Run montecarlo:/usr/bin/time -f '%e' ./run_1g_weak.sh
scale-up,MTC,weak,1,0.0
```

By installing mpich library through ```sudo apt install mpich``` and then running <em>run_scale_up.py</em>, we obtained:
```
rm -f MonteCarlo
/usr/local/cuda//bin/nvcc -arch=sm_60  -O3  -I. -I/usr/local/cuda//include -I/home/ubuntu/NVIDIA_GPU_Computing_SDK//C/common/inc -I../../common/inc/ -I/home/ubuntu/NVIDIA_GPU_Computing_SDK//shared/inc -I/home/ubuntu/opt/miniconda2/pkgs/mpich2-1.4.1p1-0//include -I/home/lian599/include/ -L. -L/home/ubuntu/NVIDIA_GPU_Computing_SDK//C/lib -L/usr/local/cuda//lib64/ -L/home/ubuntu/NVIDIA_GPU_Computing_SDK//shared/lib -L/home/ubuntu/opt/miniconda2/pkgs/mpich2-1.4.1p1-0//lib -L/home/lian599/lib/ -L/usr/lib/ -L/usr/lib64  -lcutil_x86_64 -lcuda -lmpich -lmpl -lnccl -lstdc++ -lm -I../common/inc -lcurand MonteCarloMultiGPU.cpp MonteCarlo_kernel.cu MonteCarlo_gold.cpp multithreading.cpp -o MonteCarlo
/usr/bin/ld: cannot find -lcutil_x86_64
/usr/bin/ld: cannot find -lnccl
collect2: error: ld returned 1 exit status
make: *** [Makefile:39: MonteCarlo] Error 1
$Run montecarlo:/usr/bin/time -f '%e' ./run_1g_strong.sh
scale-up,MTC,strong,1,0.0
$Run montecarlo:/usr/bin/time -f '%e' ./run_1g_weak.sh
scale-up,MTC,weak,1,0.0
```

Eventually, we notided that <em>cutil_x86_64</em> and <em>nccl</em> libraries were not used for the Montecarlo algorithm, so we decided to remove them from the makefile <em>shared.mk</em>.
So, we passed from: <br />
```NVCC_LIB = -lcutil_x86_64 -lcuda -lmpich -lmpl -lnccl```<br />
To:<br />
```NVCC_LIB = -lcuda -lmpich -lmpl``` <br />

By doing these steps, we were able to execute <em>run_scale_up.py</em>. We obtained this output:
```
MonteCarlo_gold.cpp multithreading.cpp -o MonteCarlo
$Run montecarlo:/usr/bin/time -f '%e' ./run_1g_strong.sh
scale-up,MTC,strong,1,3.058
$Run montecarlo:/usr/bin/time -f '%e' ./run_1g_weak.sh
scale-up,MTC,weak,1,3.0460000000000003
```

Insted, by running:
```
  $ cd scale-up/$app$/
  $ make
  $ chmod +x run.sh
  $ ./run.sh
```

We obtained
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

Total time (ms.): 2680.186035
        Note: This is elapsed time for all to compute.
Options per sec.: 391232.543654
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.810705E-04
Average reserve: 12.511987

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

Total time (ms.): 2689.382080
        Note: This is elapsed time for all to compute.
Options per sec.: 389894.767191
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

Total time (ms.): 2665.562012
        Note: This is elapsed time for all to compute.
Options per sec.: 393378.955504
main(): comparing Monte Carlo and Black-Scholes results...
Shutting down...
Test Summary...
L1 norm        : 4.810705E-04
Average reserve: 12.511987

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

Test passed
```


## Team
* [Andrea Paparella](https://github.com/engpap)
* [Andrea Piras](https://github.com/andreapiras00)

## Useful Commands
```conda deactivate```<br />
```conda activate``` 
