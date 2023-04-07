## From CUDA11 to Unified Memory

To port the code written in CUDA 11 to Unified Memory, we  had to make changes to memory allocation and memory access in the code. Unified Memory is a memory architecture in which CPU and GPU share the same memory space, and the memory is automatically migrated between CPU and GPU when needed. 

This simplifies the programming model, as we don't have to explicitly manage the memory allocation and data transfer between CPU and GPU.
Thus, we firstly modified the MonteCarlo_common.h file by removing variables related to the concept of device (GPU) and host (CPU) inside the plan data structure.
