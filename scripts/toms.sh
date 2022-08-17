#!/bin/bash

while getopts t:b:m:c:r: flag
do
    case "${flag}" in
        t) twitter1x_dir=${OPTARG};;
        c) cuda=${OPTARG};;
        r) reps=${OPTARG};;
    esac
done

device=cuda:$cuda

# TomVAuLT

python experiments/tmsc_tombert.py TomViLT \
    --model_name bert-base-uncased --vilt_model dandelin/vilt-b32-mlm --use_tweet_bert \
    --resnet_depth 101 --dir $twitter1x_dir/twitter2015 \
    --train_split train --dev_split dev --device $device --num_train_epochs 10 --reps $reps \
    --max_total_length 40 --max_target_length 10

python experiments/tmsc_tombert.py TomViLT \
    --model_name bert-base-uncased --vilt_model dandelin/vilt-b32-mlm --use_tweet_bert \
    --resnet_depth 101 --dir $twitter1x_dir/twitter \
    --train_split train --dev_split dev --device $device --num_train_epochs 10 --reps $reps \
    --max_total_length 40 --max_target_length 10

# TomViLT

python experiments/tmsc_tombert.py TomViLT \
    --model_name bert-base-uncased --vilt_model dandelin/vilt-b32-mlm \
    --resnet_depth 101 --dir $twitter1x_dir/twitter2015 \
    --train_split train --dev_split dev --device $device --num_train_epochs 10 --reps $reps \
    --max_total_length 40 --max_target_length 10

python experiments/tmsc_tombert.py TomViLT \
    --model_name bert-base-uncased --vilt_model dandelin/vilt-b32-mlm \
    --resnet_depth 101 --dir $twitter1x_dir/twitter \
    --train_split train --dev_split dev --device $device --num_train_epochs 10 --reps $reps \
    --max_total_length 40 --max_target_length 10