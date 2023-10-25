#!/bin/bash
file=$1

if [[ $# -eq 0 ]] ; then
    echo 'no file argument given'
    exit 0
fi

MAX_TIME=810000
IFS=$'\n'
m=$(cat $file | wc -l) # number of arguments in args.txt file
i=0

for ((i=0; i<$m; i++)); do
  line=$(sed -n "$((i+1))p" $file)
  if [[ "$line" != +(*"&"*|*"#"*) ]]; then
    sed -n "$((i+1))p" $file | xargs timeout $MAX_TIME python train.py
  fi
done
wait
