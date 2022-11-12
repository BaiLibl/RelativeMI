#!/bin/bash

file_dir=$1
mia_mt=("sd_adv" "sd_acc" "max_adv" "max_acc")
_type=("rand" "iter")

mkdir -p results

for (( i=0; i<5; i++ ))
do
    _dir=${file_dir}'_'$i
    echo '=================='$_dir'=================='
    if [ -d "${_dir}" ]; then
      for mt in "${mia_mt[@]}"
      do
        for tp in "${_type[@]}"
        do
          cp $_dir/eval_${_dir}.csv results
          python eval_models.py --dir ${_dir} --mia_mt $mt --type $tp
        done
      done
    fi
done