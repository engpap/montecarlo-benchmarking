make clean
make

export CUDA_VISIBLE_DEVICES=0,1

for scaling in weak strong
do
    for method in threaded streamed
    do
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method
        nvprof --log-file ./nvprof/nvprof_2g_x32_${scaling}_${method}_UMv2.txt ./MonteCarlo --scaling=$scaling --method=$method
        #nsys profile -o report_2g_{$scaling}_{$method}_UMv2 --stats=true --cuda-memory-usage=true --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true --force-overwrite=true ./MonteCarlo --scaling=$scaling --method=$method 
    done
done