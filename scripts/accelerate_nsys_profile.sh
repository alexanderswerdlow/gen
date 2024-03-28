#!/bin/bash

# Use $SLURM_LOCALID with srun.
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --output="%p_$timestamp" --force-overwrite=true --cudabacktrace=true --osrt-threshold=10000 -x true "$@"