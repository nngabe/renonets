#!/usr/bin/env bash
export RENONET_HOME="$(pwd)/renonet"
export NN_HOME="$RENONET_HOME/nn"
export LOG_DIR="$NN_HOME/logs"
export PYTHONPATH="$RENONET_HOME:$PYTHONPATH"
export PYTHONPATH="$NN_HOME:$PYTHONPATH"
export DATAPATH="$RENONET_HOME/data"
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
#source activate hgcn  # replace with source hgcn/bin/activate if you used a virtualenv
