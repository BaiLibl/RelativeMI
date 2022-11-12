#!/bin/bash

random_seed=$1
dataset=$2

if [ $dataset == "location" ]
then 
    L2_arr=(0.0 0.0001 0.0005 0.001 0.003 0.005 0.007 0.01)
    DR_arr=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
    epc=100
    target_name=mlp
elif [ $dataset == "purchase" ]
then
    L2_arr=(0.0 0.0001 0.0005 0.001 0.003 0.005 0.007)
    DR_arr=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7)
    epc=100
    target_name=mlp
elif [ $dataset == "stl10" ]
then
    L2_arr=(0.0 0.001 0.005 0.01)
    DR_arr=(0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)
    epc=100
    target_name=cnn
elif [ $dataset == "fmnist" ]
then
    L2_arr=(0.0 0.0005 0.001)
    DR_arr=(0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)
    epc=100
    target_name=cnn
elif [ $dataset == "cifar10" ]
then
    L2_arr=(0.0 0.0002 0.0005 0.001 0.01)
    DR_arr=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7)
    epc=100
    target_name=alex
fi


files_dir="${dataset}_${target_name}_${random_seed}"

#### train shadow model
python train_shadow.py --random_seed $random_seed --dataset $dataset --target_name $target_name 

#### train target model and mount mia attack
train () {
    l2=$1
    dropout=$2
    python pipeline.py --random_seed $random_seed \
                    --dataset $dataset \
                    --target_name $target_name \
                    --L2 $l2 \
                    --dropout $dropout \
                    --epoch $epc 

}

for wd in "${L2_arr[@]}"
do
    for drp in "${DR_arr[@]}"
    do
        echo " ================ train $wd $drp ======================"
        train $wd $drp
    done
done


#### get loss file
# python get_loss.py --dir $files_dir

#### model risk evaluate
# risk_eval () {
#     python eval_models.py --dir $1 --mia_metric $2 --type $3
# }

# risk --dir $files_dir --mia_metric 'max_acc' --type iteration
# risk --dir $files_dir --mia_metric 'max_acc' --type random
    