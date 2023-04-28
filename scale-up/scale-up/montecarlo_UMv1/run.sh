export CUDA_VISIBLE_DEVICES=0,1

make clean
make
for scaling in weak strong
do
    for method in threaded streamed
    do  # ten runs for each configuration in order to get averages
        ./MonteCarlo --scaling=$scaling --method=$method # COLD RUN
        ./MonteCarlo --scaling=$scaling --method=$method # 1
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method
        ./MonteCarlo --scaling=$scaling --method=$method # 10
        nvprof --log-file nvprof_1g_x32_{$scaling}_{$method}.txt ./MonteCarlo --scaling=$scaling --method=$method
        #nsys profile -o report_2g_x32_{$scaling}_{$method}_baseline --stats=true --cuda-memory-usage=true --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true --force-overwrite=true ./MonteCarlo --scaling=$scaling --method=$method 
    done
done
