#!/bin/bash
random_seed=$1
dataset=$2

# DEL_arr=(0.0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2)
DEL_arr=(0.0 0.025)

if [ $dataset == "location" ]
then 
    epc=100
    target_name=mlp
elif [ $dataset == "purchase" ]
then
    epc=100
    target_name=mlp
elif [ $dataset == "stl10" ]
then
    epc=100
    target_name=cnn
elif [ $dataset == "fmnist" ]
then
    epc=100
    target_name=cnn
elif [ $dataset == "cifar10" ]
then
    epc=100
    target_name=alex
fi

# train refer_model
python train_target.py --random_seed $random_seed --dataset $dataset --target_name $target_name --dropout 0.0 
python train_target.py --random_seed $random_seed --dataset $dataset --target_name $target_name --dropout 0.2

#### train shadow model
shadow_path=${dataset}_${target_name}_${random_seed}/shadow_model/shadow_model.pth
if [ ! -f "$shadow_path" ]; then
    python train_shadow.py --random_seed $random_seed --dataset $dataset --target_name $target_name 
fi

# sample removal
remo () {
    python eval_removal.py --dataset ${dataset} \
                      --target_name ${target_name} \
                      --random_seed $random_seed \
                      --risky $1 \
                      --delete $2
}

for wd in "${DEL_arr[@]}"
do
  remo 0 $wd
  remo 1 $wd
  remo 2 $wd
done
