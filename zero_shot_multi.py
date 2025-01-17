#!/usr/bin/python
import argparse
import pathlib
import string
import torch
import sys
import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import itertools
from typing import List, Tuple
from itertools import combinations
from transformers import EsmTokenizer, EsmForMaskedLM

# Credit: this code is adapted from https://github.com/amelie-iska/Variant-Effects.
# It contains several scripts for accessing variant effects (single-site and multi-site of mutations) with protein and DNA language models.
# To save computational cost, we only use wt-msarginals to infer variant effect in our study.

# Removes insertions from a sequence, needed for aligned sequences in MSA processing
def remove_insertions(sequence: str) -> str:
    # Delete lowercase characters and insertion characters ('.', '*') from the string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)

# Function to label a row in the DataFrame based on the scoring of mutations
def label_row(row, sequence, token_probs, tokenizer, offset_idx):
    # Extract wild type, index, and mutated type from the row
    mutlist = row.split('+')
    accumulated_score = 0
    for mut in mutlist:
        wt, idx, mt = mut[0], int(mut[1:-1]) - offset_idx, mut[-1]
        # print(wt, idx, mt, sequence[idx])
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
        # Encode the wild type and mutated type
        wt_encoded, mt_encoded = tokenizer.encode(wt, add_special_tokens=False)[0], tokenizer.encode(mt, add_special_tokens=False)[0]
        # Calculate the score as the difference in log probabilities
        score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
        accumulated_score += score
    score_avg = accumulated_score / len(mutlist)
    return score_avg.item()

# Function to compute pseudo-perplexity for a row, used in language model evaluation
def compute_pppl(row, sequence, model, tokenizer, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
    # Modify the sequence with the mutation
    sequence = sequence[:idx] + mt + sequence[(idx + 1) :]
    # Tokenize the modified sequence
    data = [("protein1", sequence)]
    batch_tokens = tokenizer.batch_encode_plus(data, return_tensors="pt", padding=True)["input_ids"]
    # Calculate log probabilities for each position
    log_probs = []
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = tokenizer.mask_token_id
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.to(device)).logits, dim=-1)
        log_probs.append(token_probs[0, i, batch_tokens[0, i]].item())
    return sum(log_probs)

# Function to apply corresponding mutations to a given sequence
def apply_mutations(df, seq):
    seqlist = []
    for index,row in df.iterrows():
        pos = row['positions'].split('+')
        mut = row['mutation'].split('+')
        mutated_seq = list(seq)
        for p,aa in zip(pos, mut):
            mutated_seq[int(p)-1] = aa[-1]
        seqlist.append(''.join(mutated_seq))
    return seqlist

# Function to generate a pandas DataFrame for each mutated sequence
def get_mutated_seq(seq, pool, max_comb=4):
    mutations = list(pool)
    data = []
    all_combinations = []
    ind = 0
    # for r in range(1, max_comb + 1):
    for r in tqdm(range(1, max_comb + 1), desc="Processing combinatorial mutations"):
        for comb in combinations(mutations, r):
            # Only compute N-site variants, n is default to 4; computational cost increases with more mutations
            if len(comb) > max_comb or len(comb) == 1:
                continue
            combined_positions = '+'.join([mut[1:-1] for mut in comb])
            all_combinations.append({
                'mutant': f'comb{len(comb)}_{ind+1}',
                'mutation': '+'.join(comb),
                'number_mutations': len(comb),
                'positions': combined_positions
            })
            ind += 1
    print(f"Generate {len(all_combinations)} combinations!")
    df_combinations = pd.DataFrame(all_combinations)
    df_combinations.reset_index()
    seqlist = apply_mutations(df_combinations, seq)
    df_combinations['seq'] = seqlist
    return df_combinations

# Create and configure the argument parser for command line interface
def create_parser():
    parser = argparse.ArgumentParser(
        description="Effect of multi-site mutations using ESM predictions."
    )
    # Define the arguments the script can accept
    parser.add_argument('--name', '-n', type=str, default='example', help='Task name.')
    parser.add_argument('--seq', '-s', type=str, required=True, help='Wild-type seq.')
    parser.add_argument('--plm', '-plm', type=str, default='esm2_t33_650M_UR50D', help='Name of the ESM model')                        
    parser.add_argument("--mut_pool", '-mp', type=str, help="Mutant for combination", default='')
    parser.add_argument("--max_comb", '-mc', type=int, default=4, help="Maximal sites for combination.")
    parser.add_argument("--mutation-col", '-col', type=str, default="mutant", help="Column in the deep mutational scan labeling the mutation")
    parser.add_argument("--offset-idx", type=int, default=1, help="Offset of the mutation positions")
    parser.add_argument("--scoring-strategy", type=str, default="wt-marginals", choices=["wt-marginals", "masked-marginals", "pseudo-ppl"], help="Scoring strategy to use")
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    parser.add_argument('--gpu', '-gd', type=int, default=0, help='GPU device ID.')
    parser.add_argument('--output', '-o', type=str, default='output/combinatorial', help='Output directory for prediction results.')
    parser.add_argument('--prefix', '-pf', type=str, default='esm2', help='Output prefix.')
    return parser

# Main function to orchestrate mutation scoring process
def main(args):
    os.makedirs(f'{args.output}', exist_ok=True)
    # Determine to use GPU or CPU
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.nogpu else "cpu")
    # Load the chosen ESM model and tokenizer
    model_name = args.plm
    model = EsmForMaskedLM.from_pretrained(model_name)
    for name, param in model.named_parameters():
        if 'contact_head.regression' in name:
            param.requires_grad = False
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    vocab = tokenizer.get_vocab()

    # Preprocess and encode the base sequence
    data = [("protein1", args.seq)]
    batch_tokens = tokenizer.batch_encode_plus(data, return_tensors="pt", padding=True)["input_ids"].to(device)

    mut_pool = pd.read_csv(args.mut_pool, sep='\t')[args.mutation_col].tolist()
    df = get_mutated_seq(seq=args.seq, pool=mut_pool, max_comb=args.max_comb)
    # Apply selected scoring strategy
    with torch.no_grad():
        token_probs = torch.log_softmax(model(batch_tokens).logits, dim=-1)
    df[args.plm] = df.apply(
        lambda row: label_row(row['mutation'], args.seq, token_probs, tokenizer, args.offset_idx),
        axis=1
    )
    df_sorted = df.sort_values(args.plm, ascending=False)
    # Save the scored mutations to a csv file
    df_sorted.to_csv(osp.join(args.output, f'{args.prefix}_{args.name}.csv'), index=False, sep='\t')
    # df.to_excel(osp.join(args.output, f'{args.prefix}_{args.name}_{args.scoring_strategy}.xlsx'), index=False)

# Script entry point for command-line interaction
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)

# Example usage:
# Note: this code is for multi-site variants prediction
# myseq=MQRAVSVVARLGFRLQAFPPALCRPLSCAQEVLRRT && python zero_shot_multi.py --name example  -s ${myseq} -mp output/single/top20_example.csv
