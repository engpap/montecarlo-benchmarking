VERSION='baseline'

make clean
make

for NUM_GPU in 4
do
    if [ $NUM_GPU -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES=0
    elif [ $NUM_GPU -eq 2 ]; then
        export CUDA_VISIBLE_DEVICES=0,1
    elif [ $NUM_GPU -eq 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3
    fi

    for size in 512 1024 4096 8192
    do
        for scaling in strong weak
        do
            for method in streamed threaded 
            do 
                ./MonteCarlo --scaling=$scaling --method=$method --size=$size # COLD RUN
                /usr/bin/time -f %e -o time.txt -a ./MonteCarlo --scaling=$scaling --method=$method --size=$size # for time averaging
                /usr/bin/time -f %e -o time.txt -a ./MonteCarlo --scaling=$scaling --method=$method --size=$size # for time averaging
                /usr/bin/time -f %e -o time.txt -a ./MonteCarlo --scaling=$scaling --method=$method --size=$size # for time averaging
                nvprof --log-file ./nvprof/nvprof_${NUM_GPU}gpu_size${size}x${size}_${scaling}_${method}_${VERSION}.txt ./MonteCarlo --scaling=$scaling --method=$method --size=$size
                nsys profile -o ./nsys/report_${NUM_GPU}gpu_size${size}x${size}_${scaling}_${method}_${VERSION} --stats=true --cuda-memory-usage=true --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true --force-overwrite=true ./MonteCarlo --scaling=$scaling --method=$method --size=$size
            done
        done
    done
done