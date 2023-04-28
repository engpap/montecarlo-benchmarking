export CUDA_VISIBLE_DEVICES=0,1

make clean
make

for scaling in strong
do
    for method in streamed
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
        nvprof --log-file ./nvprof/nvprof_2g_x32_${scaling}_${method}_UMv1.txt ./MonteCarlo --scaling=$scaling --method=$method
        #nsys profile -o report_2g_x32_${scaling}_${method}_UMv1 --stats=true --cuda-memory-usage=true --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true --force-overwrite=true ./MonteCarlo --scaling=$scaling --method=$method 
    done
done
