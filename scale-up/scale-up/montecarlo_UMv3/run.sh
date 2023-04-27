make clean
make
./MonteCarlo --scaling=strong --method=streamed
./MonteCarlo --scaling=strong --method=streamed
nsys profile -o report_2g_strong_streamed_UMv1 --stats=true --cuda-memory-usage=true --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true ./MonteCarlo --method=streamed --scaling=strong