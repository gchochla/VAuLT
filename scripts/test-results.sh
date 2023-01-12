#!/bin/bash

while getopts t:b:m:c:r: flag
do
    case "${flag}" in
        t) twitter1x_dir=${OPTARG};;
        b) bloomberg_dir=${OPTARG};;
        m) mvsa_dir=${OPTARG};;
        c) cuda=${OPTARG};;
        r) reps=${OPTARG};;
    esac
done

device=cuda:$cuda

# TWITTER-1X

python experiments/clsf_vault.py Twitter201X --dir $twitter1x_dir/twitter2015 \
    --vilt_model dandelin/vilt-b32-mlm  --train_split train dev --test_split test \
    --preprocess_on_fetch --device $device --num_train_epochs 15 --reps $reps

python experiments/clsf_vault.py Twitter201X --dir $twitter1x_dir/twitter \
    --vilt_model dandelin/vilt-b32-mlm --train_split train dev --test_split test \
    --preprocess_on_fetch --device $device --num_train_epochs 15 --reps $reps

python experiments/clsf_vault.py Twitter201X --dir $twitter1x_dir/twitter2015 \
    --vilt_model dandelin/vilt-b32-mlm --bert_model bert-base-uncased \
    --train_split train dev --test_split test \
    --preprocess_on_fetch --device $device --num_train_epochs 25 --reps $reps

python experiments/clsf_vault.py Twitter201X --dir $twitter1x_dir/twitter \
    --vilt_model dandelin/vilt-b32-mlm --bert_model bert-base-uncased \
    --train_split train dev --test_split test \
    --preprocess_on_fetch --device $device --num_train_epochs 25 --reps $reps

python experiments/clsf_vault.py Twitter201X --dir $twitter1x_dir/twitter2015 \
    --vilt_model dandelin/vilt-b32-mlm --bert_model vinai/bertweet-base \
    --train_split train dev --test_split test \
    --preprocess_on_fetch --device $device --num_train_epochs 15 --reps $reps

python experiments/clsf_vault.py Twitter201X --dir $twitter1x_dir/twitter \
    --vilt_model dandelin/vilt-b32-mlm --bert_model vinai/bertweet-base \
    --train_split train dev --test_split test \
    --preprocess_on_fetch --device $device --num_train_epochs 15 --reps $reps

# Bloomberg

python experiments/clsf_vault.py Bloomberg --root_dir $bloomberg_dir \
    --vilt_model dandelin/vilt-b32-mlm --train_split train dev --test_split test \
    --image_augmentation --device $device --train_batch_size 16 --num_train_epochs 8 \
    --reps $reps

python experiments/clsf_vault.py Bloomberg --root_dir $bloomberg_dir \
    --vilt_model dandelin/vilt-b32-mlm --bert_model bert-base-uncased \
    --train_split train dev --test_split test \
    --image_augmentation --device $device --train_batch_size 16 --num_train_epochs 15 \
    --reps $reps

python experiments/clsf_vault.py Bloomberg --root_dir $bloomberg_dir \
    --vilt_model dandelin/vilt-b32-mlm --bert_model vinai/bertweet-base \
    --train_split train dev --test_split test \
    --image_augmentation --device $device --train_batch_size 16 --num_train_epochs 8 \
    --reps $reps

# MVSA

python experiments/clsf_vault.py MVSA --root_dir $mvsa_dir/MVSA \
    --vilt_model dandelin/vilt-b32-mlm --train_split train dev --test_split test \
    --image_augmentation --preprocessed --device $device --train_batch_size 16 \
    --num_train_epochs 2 --reps $reps --max_num_workers 5

python experiments/clsf_vault.py MVSA --root_dir $mvsa_dir/MVSA_Single \
    --vilt_model dandelin/vilt-b32-mlm --train_split train dev --test_split test \
    --image_augmentation --preprocessed --device $device --train_batch_size 16 \
    --num_train_epochs 15 --reps $reps --max_num_workers 5

python experiments/clsf_vault.py MVSA --root_dir $mvsa_dir/MVSA \
    --vilt_model dandelin/vilt-b32-mlm --bert_model bert-base-uncased \
    --train_split train dev --test_split test \
    --image_augmentation --preprocessed --device $device --train_batch_size 16 \
    --num_train_epochs 3 --reps $reps --max_num_workers 5

python experiments/clsf_vault.py MVSA --root_dir $mvsa_dir/MVSA_Single \
    --vilt_model dandelin/vilt-b32-mlm --bert_model bert-base-uncased \
    --train_split train dev --test_split test \
    --image_augmentation --preprocessed --device $device --train_batch_size 16 \
    --num_train_epochs 15 --reps $reps --max_num_workers 5

python experiments/clsf_vault.py MVSA --root_dir $mvsa_dir/MVSA \
    --vilt_model dandelin/vilt-b32-mlm --bert_model vinai/bertweet-base \
    --train_split train dev --test_split test \
    --image_augmentation --preprocessed --device $device --train_batch_size 16 \
    --num_train_epochs 2 --reps $reps --max_num_workers 5

python experiments/clsf_vault.py MVSA --root_dir $mvsa_dir/MVSA_Single \
    --vilt_model dandelin/vilt-b32-mlm --bert_model vinai/bertweet-base \
    --train_split train dev --test_split test \
    --image_augmentation --preprocessed --device $device --train_batch_size 16 \
    --num_train_epochs 9 --reps $reps --max_num_workers 5

