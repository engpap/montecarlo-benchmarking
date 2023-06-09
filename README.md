# MonteCarlo Benchmarking
This is a project of the [High Performance Processors and Systems](https://www4.ceda.polimi.it/manifesti/manifesti/controller/ManifestoPublic.do?EVN_DETTAGLIO_RIGA_MANIFESTO=evento&aa=2022&k_cf=225&k_corso_la=481&k_indir=T2A&codDescr=089185&lang=EN&semestre=2&idGruppo=4474&idRiga=281811) course of Politecnico Di Milano.

This repository contains the evaluation and implementation of MonteCarlo workload on multi-GPU systems via Unified Memory and GrCUDA framework.
The project is based on the [Tartan benchmarking suite](https://github.com/uuudown/Tartan/blob/master/IISWC-18.pdf), more specifically on the MonteCarlo benchmark scale-up implementation.

## Team
* [Andrea Paparella](https://github.com/engpap)
* [Andrea Piras](https://github.com/andreapiras00)

## Repository Structure
Below is a brief overview of the main directories and files:

### common/
A directory for common resources used across the project.

### montecarlo/ 
A directory dedicated to the GrCUDA implementation of the MonteCarlo benchmark.

### scale-up/ 
A directory dedicated to the CUDA implementation of the MonteCarlo benchmark. It contains different versions:
* /montecarlo: CUDA baseline version which reflects [Tartan implementation](https://github.com/uuudown/Tartan/blob/master/), ported in CUDA 11
* /montecarlo_UM: Unified Memory implementation
* /montecarlo_UMv1: Unified Memory implementation with prefetching
* /montecarlo_UMv2: Unified Memory implementation with advising
* /montecarlo_UMv3: Unified Memory implementation with prefetching & advising

### IISWC-18.pdf 
Paper of the project's benchmark suite.

### grcuda-0.1.1.jar 
The grcuda jar file used for implementing the GrCUDA version.

### plot.py
A Python script used for creating plots from profiling data.

### run_grcuda.sh 
A shell script used to run grcuda with all different configurations.

### shared.mk
Shared Makefile for the project.

## How to run the benchmark versions


## Useful Commands
```conda deactivate```<br />
```conda activate``` <br />
<br />

Nvprof:<br />

```nvprof ./MonteCarlo --method=<method> --scaling=<scaling>```<br />

To print the gpu trace: <br />

```nvprof --print-gpu-trace ./MonteCarlo --method=<method> --scaling=<scaling>```<br />


Nsight Systems:<br />

```nsys profile -o <report_file_name> --stats=true --cuda-memory-usage=true --cuda-um-cpu-page-faults=true 	--cuda-um-gpu-page-faults=true --force-overwrite=true ./MonteCarlo --method=<method> --scaling=<scaling>```<br />

```--stats=true``` to generate a summary of CPU and GPU activities
or<br />
```nsys profile ./MonteCarlo```<br />

###========================================================================
####         Author:  Ang Li, PNNL
####        Website:  http://www.angliphd.com  
####        Created:  03/19/2018 02:44:40 PM, Richland, WA, USA.
###========================================================================

- Introduction:
 This directory contains the 7 multi-GPU benchmarks for intr-node scale-up with PCI-e 
 and NVLink-V1/V2 interconnect.

- Config:
 Set env in "shared.mk". You need to install NCCL library before building the NVLink version.

- Run:
 Execute the python script: 
 ```
   $ python run_scale_up.py 
 ```
 or enter into each app dir, make, and run:
 ```
   $ cd scale-up/$app$/
   $ make
   $ chmod +x run.sh
   $ ./run.sh
 ```
##Scaling Test:
  Strong scaling:
  ```
   $ ./run_1g_strong.sh
   $ ./run_2g_strong.sh
   $ ./run_4g_strong.sh
   $ ./run_8g_strong.sh
  ```
  
  Weak scaling: 
  ```
   $ ./run_1g_weak.sh
   $ ./run_2g_weak.sh
   $ ./run_4g_weak.sh
   $ ./run_8g_weak.sh
  ```
##File Description: 
```shell
            shared.mk: Overall configuration file for all Makefile. Please config your env here.

               common: Commonly-shared header and/or dependent third-party library

             scale-up: Benchmarks based on PCI-e interconnect

      scale-up-nvlink: Benchmarks based on NVLink interconnect

      run_scale_up.py: Python script for testing

```

##Applications:
```shell
            ConvNet2: Convolution neural networks via data, model and hybrid parallelism

            Cusimann: global optmization via parallel simulated annealing algorithm.

                 GMM: multivariate data clustering via Gaussian mixture model

              Kmeans: Kmeans-Clustering for double-precision data.

          MonteCarlo: Monte-Carlo option pricing from CUDA-SDK

              Planar: Depth-First-Search (DFS) and backtracing to solve Planar Langford's Sequence

              Trueke: Exchange Monte-Carlo for 3D random field Ising model

```


## Note:

    Please see our IISWC-18 paper "Tartan: Evaluating Modern GPU Interconnect via a 
      Multi-GPU Benchmark Suite" for detail.

    Please cite our paper if you find this package useful.  
    


