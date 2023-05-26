#include <cuda_runtime.h>
#include <helper_cuda.h>

/// Utility function to tweak problem size for small GPUs
int adjustProblemSize(int GPU_N, int default_nOptions)
{
    int nOptions = default_nOptions;

    // select problem size
    for (int i=0; i<GPU_N; i++)
    {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, i));
        int cudaCores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)
                        * deviceProp.multiProcessorCount;

        if (cudaCores <= 32)
        {
            nOptions = (nOptions < cudaCores/2 ? nOptions : cudaCores/2);
        }
    }

    return nOptions;
}

int adjustGridSize(int GPUIndex, int defaultGridSize)
{
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, GPUIndex));
    int maxGridSize = deviceProp.multiProcessorCount * 40;
    return ((defaultGridSize > maxGridSize) ? maxGridSize : defaultGridSize);
}
