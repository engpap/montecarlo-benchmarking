#!/bin/bash

num_elems=(512 1024 4096 8192)
num_blocks=(1024)
choose_device_policies=("round-robin")
scaling_choices=("strong" "weak")
prefecth_values=(true false)
num_gpus=(4)
block_sizes1d=(256)

PATH_TO_CONFIG_FILE="/home/ubuntu/montecarlo-benchmarking/montecarlo/src/test/java/com/polimi/montecarlo/config_V100.json"

for num_elem in ${num_elems[@]}; do
    for num_block in ${num_blocks[@]}; do
        for choose_device_policy in ${choose_device_policies[@]}; do
            for scaling_choice in ${scaling_choices[@]}; do
                for prefetch_val in ${prefecth_values[@]}; do
                    for num_gpu in ${num_gpus[@]}; do
                        for block_size1d in ${block_sizes1d[@]}; do
                            # Using jq to manipulate JSON file
                            #!/bin/bash
                            jq -n \
                            --argjson num_elem $num_elem \
                            --argjson num_block $num_block \
                            --arg choose_device_policy $choose_device_policy \
                            --arg scaling_choice $scaling_choice \
                            --argjson prefecth_val $prefetch_val \
                            --arg num_gpu $num_gpu \
                            --argjson block_size1d $block_size1d \
                            '.num_iter = 1 | .reAlloc = true | .reInit = true | .randomInit = false | .cpuValidation = true | .heap_size = 470 | .debug = true | .nvprof_profile = false | .num_elem.MonteCarlo[0] = $num_elem | .benchmarks[0] = "MonteCarlo" | .numBlocks.MonteCarlo = $num_block | .exec_policies[0] = "async" | .dependency_policies[0] = "with-const" | .new_stream_policies[0] = "always-new" | .parent_stream_policies[0] = "disjoint" | .choose_device_policies[0] = $choose_device_policy | .memory_advise[0] = "none" | .scalingChoice[0] = $scaling_choice | .prefetch[0] = $prefecth_val | .stream_attach[0] = false | .time_computation[0] = false | .num_gpus[0] = $num_gpu | .block_size1d.MonteCarlo = $block_size1d' \
                            ... > $PATH_TO_CONFIG_FILE  # Direct jq output to config file
                            for i in {1..6}; do
                                    cd /home/ubuntu/montecarlo-benchmarking/montecarlo/ && mvn test
                                done

                            cd /home/ubuntu/montecarlo-benchmarking/montecarlo/

                            nvprof --print-gpu-trace --profile-child-processes --log-file ./nvprof/nvprof_${num_gpu}gpu_size${num_elems}x${num_elems}_${scaling_choice}_prefetch${prefetch_val}_%p.txt mvn test

                            nsys profile -o ./nsys/report_${num_gpu}gpu_size${num_elems}x${num_elems}_${scaling_choice}_prefetch${prefetch_val} --stats=true --cuda-memory-usage=true --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true --force-overwrite=true mvn test
                        done
                    done
                done
            done
        done
    done
done


'''
The issue is that the --log-file option for nvprof is set to use a specific filename, 
but nvprof is trying to profile multiple processes that all try to write to the same file, 
causing a conflict. By appending "%p" (which represents the process ID) to the log file name, 
each process will write to a unique file and avoid conflicts.
'''
