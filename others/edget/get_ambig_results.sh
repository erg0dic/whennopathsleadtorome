#!/bin/bash
python train.py --data_path ambig --dataset_type ambiguity --exp_name ambig_no_input_rat --input_rep other --model_type rat
python train.py --data_path ambig --dataset_type no_ambiguity --exp_name no_ambig_no_input_rat --input_rep other --model_type rat
python train.py --data_path ambig --dataset_type ambiguity --exp_name ambig_input_rat --model_type rat
python train.py --data_path ambig --dataset_type no_ambiguity --exp_name no_ambig_input_rat --model_type rat

python train.py --data_path ambig --dataset_type ambiguity --exp_name ambig_no_input --input_rep other
python train.py --data_path ambig --dataset_type no_ambiguity --exp_name no_ambig_no_input --input_rep other
python train.py --data_path ambig --dataset_type ambiguity --exp_name ambig_input
python train.py --data_path ambig --dataset_type no_ambiguity --exp_name no_ambig_input
