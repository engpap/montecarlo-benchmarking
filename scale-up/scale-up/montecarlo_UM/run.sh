export CUDA_VISIBLE_DEVICES=0,1

make clean
make
for scaling in strong #weak
do
    for method in streamed #threaded 
    do  # ten runs for each configuration in order to get averages
        ./MonteCarlo --scaling=$scaling --method=$method # COLD RUN
        /usr/bin/time -f %e -o time.txt -a ./MonteCarlo --scaling=$scaling --method=$method # 1
        /usr/bin/time -f %e -o time.txt -a ./MonteCarlo --scaling=$scaling --method=$method
        /usr/bin/time -f %e -o time.txt -a ./MonteCarlo --scaling=$scaling --method=$method
        /usr/bin/time -f %e -o time.txt -a ./MonteCarlo --scaling=$scaling --method=$method
        /usr/bin/time -f %e -o time.txt -a ./MonteCarlo --scaling=$scaling --method=$method
        /usr/bin/time -f %e -o time.txt -a ./MonteCarlo --scaling=$scaling --method=$method
#        ./MonteCarlo --scaling=$scaling --method=$method
#        ./MonteCarlo --scaling=$scaling --method=$method
#        ./MonteCarlo --scaling=$scaling --method=$method
#        ./MonteCarlo --scaling=$scaling --method=$method
#        ./MonteCarlo --scaling=$scaling --method=$method
#        ./MonteCarlo --scaling=$scaling --method=$method
#        nvprof --log-file ./nvprof/nvprof_2g_x32_${scaling}_${method}_baseline.txt ./MonteCarlo --scaling=$scaling --method=$method
        #nsys profile -o report_2g_x32_${scaling}_${method}_baseline --stats=true --cuda-memory-usage=true --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true --force-overwrite=true ./MonteCarlo --scaling=$scaling --method=$method 
    done
done