#!/bin/bash

random_seed=$1
dataset=$2
epc=100

#### train target model and mount mia attack
train () {
    l2=$1
    dropout=$2
    net=$3
    python pipeline.py --random_seed $random_seed \
                    --dataset $dataset \
                    --target_name $net \
                    --L2 $l2 \
                    --dropout $dropout \
                    --epoch $epc

}

train_loop () {
    L2_vec=$1
    DR_vec=$2
    network=$3
    for wd in ${L2_vec[*]}
    do
        for drp in ${DR_vec[*]}
        do
            echo " ================ train $wd $drp ======================"
            train $wd $drp $network
            # echo $wd $drp $network
        done
    done
}

#
if [ $dataset == "stl10" ]
then
    #### train shadow model
    target_name=cnn
    date
    python train_shadow.py --random_seed $random_seed --dataset $dataset --target_name $target_name
    date
    L2_arr=(0.0 0.001 0.01)
    DR_arr=(0.0 0.1 0.2 0.3 0.4 0.5)
    train_loop "${L2_arr[*]}" "${DR_arr[*]}" $target_name

    target_name=alex
    date
    python train_shadow.py --random_seed $random_seed --dataset $dataset --target_name $target_name
    date
    L2_arr=(0.0 0.001 0.01)
    DR_arr=(0.0 0.1 0.2 0.3 0.4 0.5)
    train_loop "${L2_arr[*]}" "${DR_arr[*]}" $target_name

    target_name=densenet
    date
    python train_shadow.py --random_seed $random_seed --dataset $dataset --target_name $target_name
    date
    L2_arr=(0.0 0.001 0.01)
    DR_arr=(0.0 0.1 0.2 0.3 0.4 0.5)
    train_loop "${L2_arr[*]}" "${DR_arr[*]}" $target_name

elif [ $dataset == "fmnist" ]
then

    #### train shadow model
    target_name=cnn
    date
    python train_shadow.py --random_seed $random_seed --dataset $dataset --target_name $target_name
    date
    L2_arr=(0.0 0.001 0.01)
    DR_arr=(0.0 0.1 0.2 0.3 0.4 0.5)
    train_loop "${L2_arr[*]}" "${DR_arr[*]}" $target_name

    target_name=alex
    date
    python train_shadow.py --random_seed $random_seed --dataset $dataset --target_name $target_name
    date
    L2_arr=(0.0 0.001 0.01)
    DR_arr=(0.0 0.1 0.2 0.3 0.4 0.5)
    train_loop "${L2_arr[*]}" "${DR_arr[*]}" $target_name

    target_name=resnet18
    date
    python train_shadow.py --random_seed $random_seed --dataset $dataset --target_name $target_name
    date
    L2_arr=(0.0 0.0005 0.001 0.005 0.01)
    DR_arr=(0.0)
    train_loop "${L2_arr[*]}" "${DR_arr[*]}" $target_name
fi
  