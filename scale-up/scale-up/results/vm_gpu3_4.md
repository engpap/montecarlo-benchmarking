## From CUDA9 to CUDA11
For carrying out the porting from CUDA 9 to CUDA 11 on a 4 GPUs virtual machine, we switched to AD2 and created a new boot volume. We changed the version of CUDA by simply changing the CUDA directory in the makefile, from: <br />
```CUDA_DIR = /usr/local/cuda-9.1/```<br />
To:<br />
```CUDA_DIR = /usr/local/cuda-11.7/``` <br />
We were able to run the program with CUDA11.


Running Montecarlo with CUDA11 (version 11.7) produces an error indicating that the linker is unable to find several libraries that are required by the program. Specifically, the libraries <em>mpich</em> and <em>mpl</em> cannot be found. In detail, we obtained the following error:
```
```
