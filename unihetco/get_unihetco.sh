#!/bin/bash
python get_unihetco.py --problem maxcut --batch_size 1 --lambda_k 1.0 --lambda_f 1.0 --num_graph_layers 2 --num_qc_layers 1 --num_ab_layers 2 --n_updates 0 --seed 42 --num_workers 4 --exp_name final_model

python get_unihetco.py --problem mis --batch_size 1 --lambda_k 1.0 --lambda_f 1.0 --num_graph_layers 2 --num_qc_layers 1 --num_ab_layers 2 --n_updates 0 --seed 42 --num_workers 4 --exp_name final_model

python get_unihetco.py --problem mvc --batch_size 1 --lambda_k 1.0 --lambda_f 1.0 --num_graph_layers 2 --num_qc_layers 1 --num_ab_layers 2 --n_updates 0 --seed 42 --num_workers 4 --exp_name final_model

python get_unihetco.py --problem maxclique --batch_size 1 --lambda_k 1.0 --lambda_f 1.0 --num_graph_layers 2 --num_qc_layers 1 --num_ab_layers 2 --n_updates 0 --seed 42 --num_workers 4 --exp_name final_model