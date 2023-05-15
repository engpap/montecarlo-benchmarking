export CUDA_VISIBLE_DEVICES=0,1
nvprof ./MonteCarlo --scaling=strong --method=threaded
