#!/usr/bin/env bash
export COSYNN_HOME=$(pwd)
export NN_HOME="$COSYNN_HOME/nn"
export LOG_DIR="$NN_HOME/logs"
export PYTHONPATH="$COSYNN_HOME:$PYTHONPATH"
export PYTHONPATH="$NN_HOME:$PYTHONPATH"
export DATAPATH="$COSYNN_HOME/data"
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
#source activate hgcn  # replace with source hgcn/bin/activate if you used a virtualenv
