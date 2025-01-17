import argparse
import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
from itertools import product


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Effect of multi-site mutations using ESM predictions."
    )
    # Define the arguments the script can accept
    parser.add_argument('--name', '-n', type=str, default='example', help='Task name.')
    parser.add_argument('--input', '-i', type=str, default='output/single/fitness_landscape_esm2_example.py', help='Output prefix.')
    parser.add_argument('--output', '-o', type=str, default='plot', help='Output directory for plots.')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    logits_col = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
       'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    residues = [a for a in logits_col]

    my_df = pd.read_csv(args.input, sep='\t')
    print(my_df.columns)
    mat = my_df[logits_col].to_numpy().transpose()
    fig, ax = plt.subplots()
    ax.matshow(mat, cmap=plt.cm.Oranges)
    cax = ax.matshow(mat, interpolation='nearest', cmap='GnBu')
    fig.colorbar(cax,fraction=0.025, pad=0.05)
    ax.set_xlim(0, mat.shape[-1]-1)
    ax.set_ylim(0, 19)
    # ax.set_xticks(np.arange(max_val))
    ax.set_yticks(range(20), residues)
    ax.set_xlabel('Sequence', fontsize=14, labelpad=10)
    ax.set_ylabel('Amino acid', fontsize=14, labelpad=10)
    ax.set_title('Model probability',fontsize=16, pad=10)
    plt.savefig(f'plot/fitness_landscape_{args.name}.png',dpi=300)
