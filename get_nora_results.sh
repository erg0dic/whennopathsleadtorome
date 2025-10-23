#!/bin/bash
cd src || exit 1
# EpiGNN
# epignn margin vanilla
python train.py experiments=fb_model_ambig experiments.exp_name=epignn_no_ambig_vanilla experiments.aggr_type=min experiments.dataset_type=no_ambiguity experiments.data_train_path='../data/ambig/train_no_ambig.csv' experiments.load_from_checkpoint=False
python train.py experiments=fb_model_ambig experiments.exp_name=epignn_ambig_vanilla experiments.aggr_type=min experiments.dataset_type=ambiguity experiments.data_train_path='../data/ambig/train_ambig.csv' experiments.load_from_checkpoint=False

python train.py experiments=fb_model_ambig experiments.exp_name=epignn_ambig_bce_min experiments.aggr_type=min experiments.dataset_type=ambiguity experiments.data_train_path='../data/ambig/train_ambig.csv' experiments.load_from_checkpoint=False experiments.use_margin_loss_multi=False
python train.py experiments=fb_model_ambig experiments.exp_name=epignn_no_ambig_bce_min experiments.aggr_type=min experiments.dataset_type=no_ambiguity experiments.data_train_path='../data/ambig/train_no_ambig.csv' experiments.load_from_checkpoint=False experiments.use_margin_loss_multi=False

python train.py experiments=fb_model_ambig experiments.exp_name=epignn_ambig_bce_mul experiments.aggr_type=mul experiments.dataset_type=ambiguity experiments.data_train_path='../data/ambig/train_ambig.csv' experiments.load_from_checkpoint=False experiments.use_margin_loss_multi=False
python train.py experiments=fb_model_ambig experiments.exp_name=epignn_no_ambig_bce_mul experiments.aggr_type=mul experiments.dataset_type=no_ambiguity experiments.data_train_path='../data/ambig/train_no_ambig.csv' experiments.load_from_checkpoint=False experiments.use_margin_loss_multi=False


cd ../others/edget || exit 1

# RAT
python train.py --data_path ambig --dataset_type ambiguity --exp_name ambig_no_input_rat --input_rep other --model_type rat
python train.py --data_path ambig --dataset_type no_ambiguity --exp_name no_ambig_no_input_rat --input_rep other --model_type rat
python train.py --data_path ambig --dataset_type ambiguity --exp_name ambig_input_rat --model_type rat
python train.py --data_path ambig --dataset_type no_ambiguity --exp_name no_ambig_input_rat --model_type rat

# ET
python train.py --data_path ambig --dataset_type ambiguity --exp_name ambig_no_input --input_rep other
python train.py --data_path ambig --dataset_type no_ambiguity --exp_name no_ambig_no_input --input_rep other
python train.py --data_path ambig --dataset_type ambiguity --exp_name ambig_input
python train.py --data_path ambig --dataset_type no_ambiguity --exp_name no_ambig_input


# RGCN
python rgcn.py --dataset_type ambiguity
python rgcn.py --dataset_type no_ambiguity

cd ../nbfnet || exit 1

# NBFNet
python model.py --dataset_type ambiguity --exp_tag ambig_nbf_vanilla
python model.py --dataset_type no_ambiguity --exp_tag no_ambig_nbf_vanilla

python model_margin.py --dataset_type ambiguity --exp_tag ambig_nbf_margin
python model_margin.py --dataset_type no_ambiguity --exp_tag no_ambig_nbf_margin
