## From CUDA11 to Unified Memory

To port the code written in CUDA 11 to Unified Memory, we  had to make changes to memory allocation and memory access in the code. Unified Memory is a memory architecture in which CPU and GPU share the same memory space, and the memory is automatically migrated between CPU and GPU when needed. 

This simplifies the programming model, as we don't have to explicitly manage the memory allocation and data transfer between CPU and GPU.
Thus, we firstly modified the MonteCarlo_common.h file by removing variables related to the concept of device (GPU) and host (CPU) inside the plan data structure.

Firstly, we modified the data structure TOptionPlan inside MonteCarlo_common.h file. From this:
![alt text](https://github.com/engpap/montecarlo-benchmarking/blob/porting-um/scale-up/scale-up/report/assets/common_cuda.jpg?raw=true)

We obtained the following:
![alt text](https://github.com/engpap/montecarlo-benchmarking/blob/porting-um/scale-up/scale-up/report/assets/common_um.jpg?raw=true)

 The first code block contains pointers to both host and device memory, along with temporary host-side pinned memory for asynchronous and faster data transfers. In contrast, the second code block employs the use of Unified Memory (UM), which simplifies memory management by automatically migrating data between host and device memory as necessary.
 The UM implementation introduces a two member variables:
 - um_CallValue, which represents the two device and host side variables destinated to store the option data, i.e. d_OptionData and h_OptionData respectively.
 - um_CallValue,  which represents the two device and host side variables destinated to store the call value, i.e. d_CallValue and h_CallValue respectively.

Secondly, we modified multiple fucntions insde the Montecarlo_kernel.cu file.

We removed the cudaMalloc and cudaMallocHost from CUDA code and replaced them with cudaMallocManged.
The reason why we did this is because cudaMalloc allocates memory on the GPU device and cudaMallocHost function is used to allocate memory on the CPU host, but we are dealing with the concept of Unified Memory. Using cudaMallocManaged simplifies the memory management in CUDA programs and eliminate the need to manually copy data between the CPU and GPU. This allows a programmer to allocate memory that can be accessed by both the CPU and GPU.

Below snippets of the code are reported.
![alt text](https://github.com/engpap/montecarlo-benchmarking/blob/porting-um/scale-up/scale-up/report/assets/initMonteCarloGPU_cuda.jpg?raw=true)
![alt text](https://github.com/engpap/montecarlo-benchmarking/blob/porting-um/scale-up/scale-up/report/assets/initMonteCarloGPU_um.jpg?raw=true)

Notice that option data and call value variables are changed accordingly. There are no longer two different variables for each value (one for host and one for device), but a single UM variable.

In the same fashion, we modified the closeMonteCarloGPU function.
Below snippets of the code are reported.
![alt text](https://github.com/engpap/montecarlo-benchmarking/blob/porting-um/scale-up/scale-up/report/assets/closeMonteCarloGPU_cuda.jpg?raw=true)
![alt text](https://github.com/engpap/montecarlo-benchmarking/blob/porting-um/scale-up/scale-up/report/assets/closeMonteCarloGPU_um.jpg?raw=true)


Something more interesting can be found in the MonteCarloGPU function. The option data is copied from the host memory to the device memory using the cudaMemcpyAsync function. Then, the MonteCarloOneBlockPerOption kernel is launched. Eventually, the call values are copied back from the device memory to the host memory.
This process can be semplified by the introduction of UM: calls to cudaMemcpyAsync function can be removed.
This function copies data between host and device, thus it is not no longer necessary. By removing it, the having two distinct variables for both the call value and the option data turned out to be useless. Thus, we modified the code and used only um_CallValue and um_OptionData.

Below snippets of the code are reported.
![alt text](https://github.com/engpap/montecarlo-benchmarking/blob/porting-um/scale-up/scale-up/report/assets/MonteCarloGPU_cuda.jpg?raw=true)
![alt text](https://github.com/engpap/montecarlo-benchmarking/blob/porting-um/scale-up/scale-up/report/assets/MonteCarloGPU_um.jpg?raw=true)

Finally, we modified the name of d_OptionData and d_CallValue to optionData and callValue, in MonteCarloOneBlockPerOption function, just for clarity.




