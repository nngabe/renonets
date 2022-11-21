#!/usr/bin/env bash
export HPGN_HOME=$(pwd)
export HGCN_HOME="$HPGN_HOME/hgcn"
export LOG_DIR="$HGCN_HOME/logs"
export PYTHONPATH="$HPGN_HOME:$PYTHONPATH"
export PYTHONPATH="$HGCN_HOME:$PYTHONPATH"
export DATAPATH="$HPGN_HOME/data"
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
#source activate hgcn  # replace with source hgcn/bin/activate if you used a virtualenv
