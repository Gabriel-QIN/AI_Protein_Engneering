#!/usr/bin/bash
# Please first download the ESM model into your local machine
# https://huggingface.co/facebook/esm2_t33_650M_UR50D

# Define your wild-type sequence
myseq=MQRAVSVVARLGFRLQAFPPALCRPLSCAQEVLRRTPLY

# Inferring the variant effect of single-site mutations
python zero_shot_single.py --name example  -s ${myseq} -o output/single

# Visulize the fitness landscape
python heatmap.py --name example -i output/single/fitness_landscape_esm2_example.py

# Inferring the variant effect of multi-site mutations given the highest-probability mutants in this previous round
python zero_shot_multi.py --name example  --seq ${myseq} --mut_pool output/single/top20_example.csv --max_comb 5