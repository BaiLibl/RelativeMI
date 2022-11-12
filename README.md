## About
This is the demo code for RMI measurement.

## Environment
The current software is tested with Pytorch 1.12.1 and Python 3.9.

## Dataset
For FMNIST and STL10, PyTorch has offered the datasets and they can be easily employed.

Download [Purchase](https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz) and [Location](https://github.com/inspire-group/membership-inference-evaluation/tree/master/MemGuard/data)

## Usage
File **MIA_shadow.py** contains the training-based MIA code, while file **MIA_metric.py** includes the metric-based MIAs.

File **pipeline.py** contains the code to train a model and mount MIAs.

`python pipeline.py --random_seed $seed --dataset $dataset --target_name $model_type --L2 $l2 --dropout $dropout --epoch $epc`


File **eval_models.py** contains the code to measure MIA risks given a set of target models.

`python eval_models.py --dir $dir --mia_mt $mt --type $type`

The folder *dir* includes folders *target_model/* and *loss/*, *mia_mt* option [max_acc, sd_acc, and so on], *type* option [iter or rand]


Folder `scripts` includes shell scripts to deal with multiple models training and evaluatation.

## Reference
[ML-DOCTOR: Holistic Risk Assessment of Inference Attacks Against Machine Learning Models](https://github.com/liuyugeng/ML-Doctor)

[Systematic Evaluation of Privacy Risks of Machine Learning Models](https://github.com/inspire-group/membership-inference-evaluation)

