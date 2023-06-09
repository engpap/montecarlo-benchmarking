# MonteCarlo Benchmarking
## Introduction
This is a project of the [High Performance Processors and Systems](https://www4.ceda.polimi.it/manifesti/manifesti/controller/ManifestoPublic.do?EVN_DETTAGLIO_RIGA_MANIFESTO=evento&aa=2022&k_cf=225&k_corso_la=481&k_indir=T2A&codDescr=089185&lang=EN&semestre=2&idGruppo=4474&idRiga=281811) course of Politecnico Di Milano.
This repository contains the evaluation and implementation of MonteCarlo workload on multi-GPU systems via Unified Memory and GrCUDA framework.
The project is based on the [Tartan benchmarking suite](https://github.com/uuudown/Tartan/blob/master/IISWC-18.pdf), more specifically on the MonteCarlo benchmark scale-up implementation.

## Technical Report
The technical report of the project can be found [here]().
The report is a comprehensive document detailing the study and implementation of Monte-Carlo simulation for option pricing, implemented using multi-GPU CUDA C++, CUDA 11 with Unified Memory, and GrCUDA with GraalVM Java.

The report covers the following aspects:
1. An introduction and an overview of the Monte-Carlo Simulation and Option Pricing, including a problem statement and research objectives.
2. A detailed description of the original multi-GPU CUDA C++ implementation, and the benchmark program flow and manual handling methods.
3. A section on the Unified Memory (UM) implementation including porting to CUDA 11, initial UM porting, and a number of optimizations such as prefetching the memory and advising memory migration.
4. An exploration of GrCUDA and GraalVM Java implementation, including porting to Java, program structure, program design, and a brief description of the classes involved. It also covers the binding from PTX files and additional optimizations.
5. A thorough methodology section that outlines performance evaluation metrics, data collection and analysis tools, and the testing procedure.
6. Results and discussion segment that compares performance across different versions, validation of data transfer patterns, profiling outputs, and various timing graphs.

Please refer to the report for a detailed understanding of the work done on this project.

# Repository Structure
Below is a brief overview of the main directories and files:

### common/
A directory for common resources used across the project.

### montecarlo/ 
A directory dedicated to the GrCUDA implementation of the MonteCarlo benchmark.

### scale-up/ 
A directory dedicated to the CUDA implementation of the MonteCarlo benchmark. It contains different versions:
* /montecarlo: CUDA baseline version which reflects [Tartan implementation](https://github.com/uuudown/Tartan/tree/master/scale-up/scale-up/montecarlo), ported in CUDA 11
* /montecarlo_UM: Unified Memory implementation
* /montecarlo_UMv1: Unified Memory implementation with prefetching
* /montecarlo_UMv2: Unified Memory implementation with advising
* /montecarlo_UMv3: Unified Memory implementation with prefetching & advising

### IISWC-18.pdf 
Paper of the project's benchmark suite.

### grcuda-0.1.1.jar 
The grcuda jar file used for implementing the GrCUDA version.


# How to set up your machine
- Configuration:
 Set env in "shared.mk". You need to install NCCL library before building the NVLink version.
- GrCUDA Installation:
 Follow the instructions in the [Nects Lab's GrCUDA repository](https://github.com/necst/grcuda#installation).
 
 # How to run the benchmarks

## Method A
To run every possible configuration, take timings, produces nvprof and Nsight Systems report files:

* For CUDA baseline version:
 ``` 
   $ cd scale-up/montecarlo/
   $ chmod +x run.sh
   $ ./run.sh
 ```

 * For CUDA version 0:
 ``` 
   $ cd scale-up/montecarlo_UM/
   $ chmod +x run.sh
   $ ./run.sh
 ```

* For CUDA version 1:
 ``` 
   $ cd scale-up/montecarlo_UMv1/
   $ chmod +x run.sh
   $ ./run.sh
 ```

 * For CUDA version 2:
 ``` 
   $ cd scale-up/montecarlo_UMv2/
   $ chmod +x run.sh
   $ ./run.sh
 ```

 * For CUDA version 3:
 ``` 
   $ cd scale-up/montecarlo_UMv3/
   $ chmod +x run.sh
   $ ./run.sh
 ```

 * For GrCUDA version:
 ``` 
   $ cd montecarlo/
   $ chmod +x run_grcuda.sh
   $ ./run_grcuda.sh
 ```

## Method B
Alternatively, you may not be interested in report file. In that case, just run the single version by entering into each app dir, make, and run:
 
* For CUDA versions: where NUM is in {1,2,4,8}, scalingChoice is in {strong, weak}, app is in {montecarlo, montecarlo_UM, montecarlo_UMv1, montecarlo_UMv2, montecarlo_UMv3}.
 ```
   $ cd scale-up/<app>/
   $ make
   $ chmod +x run.sh
   $ ./run_<NUM>g_<scalingChoice>.sh
 ```

* For GrCUDA version:
```
  $ cd montecarlo/src/test/java/com/polimi/montecarlo
  $ mvn test
```

# Useful Commands

### Conda<br />
```conda deactivate```<br />
```conda activate``` <br />

### Nvprof:<br />
```nvprof ./MonteCarlo --method=<method> --scaling=<scaling>```<br />
To print the gpu trace: <br />
```nvprof --print-gpu-trace ./MonteCarlo --method=<method> --scaling=<scaling>```<br />

### Nsight Systems:<br />
```nsys profile -o <report_file_name> --stats=true --cuda-memory-usage=true --cuda-um-cpu-page-faults=true 	--cuda-um-gpu-page-faults=true --force-overwrite=true ./MonteCarlo --method=<method> --scaling=<scaling>```<br />
To generate a summary of CPU and GPU activities: <br />
```--stats=true``` 
or<br />
```nsys profile ./MonteCarlo```<br />