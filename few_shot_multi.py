#!/usr/bin/python
# Credit: thanks for https://github.com/amelie-iska/Variant-Effects
# Credit: thanks for https://github.com/ntranoslab/esm-variants

import re
import os, gc
import os.path as osp
import numpy as np
import pandas as pd
import numpy as np
import torch
import esm
import argparse
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax 
from scipy.stats import entropy
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def return_amino_acid_df(df):    
    # Feature Engineering on Train Data
    search_amino=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for amino_acid in search_amino:
         df[amino_acid]=df['protein_sequence'].str.count(amino_acid,re.I)
    return df

def extract_wt_features(model, batch_converter, device, data, tokens):
    # Ensure model is on the correct device
    model = model.to(device)
    
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # Move batch_tokens to the same device as the model
    batch_tokens = batch_tokens.to(device)
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    # Generate per-sequence representations via averaging
    sequence_representations = []
    for i, (_, seq) in enumerate(data):
        sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))
    # Extract attention contact maps for each sequence
    attention_contacts_list = []
    for (_, seq), attention_contacts in zip(data, results["contacts"]):
        attention_contacts_list.append(attention_contacts[: len(seq), : len(seq)].cpu())
    # Predict mutations
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[], return_contacts=False)
    # Convert logits to numpy
    logits_np = results['logits'].detach().cpu().numpy()
    # Apply softmax
    probs = softmax(logits_np, axis=-1)
    # Create an empty matrix for the probabilities
    probs_matrix = np.zeros((len(tokens), len(data[0][1])))
    # Populate the matrix with probabilities
    for i, token in enumerate(tokens):
        if token in alphabet.tok_to_idx:
            index = alphabet.tok_to_idx[token]
            probs_matrix[i] = probs[0, 1:-1, index]
    return attention_contacts_list, sequence_representations, probs_matrix

def plot_attention_contacts(attention_contacts_list, data, name=''):
    for (_, seq), attention_contacts in zip(data, attention_contacts_list):
        plt.figure(figsize=(10, 10))  # adjust as needed
        plt.imshow(attention_contacts.numpy())
        plt.title(f'Attention Contacts for Sequence WildType')
        plt.colorbar(label='Contact Score')
        plt.xlabel('Residue Position')
        plt.ylabel('Residue Position')
    plt.savefig(f'plot/{name}_attention_contact.png', dpi=600)

def plot_attention_contacts_bw(attention_contacts_list, data, name=''):
    for (_, seq), attention_contacts in zip(data, attention_contacts_list):
        plt.figure(figsize=(10, 10))
        plt.imshow(attention_contacts.numpy(), cmap='binary')
        plt.title(f'Attention Contacts for Sequence WildType')
        plt.colorbar(label='Contact Score')
        plt.xlabel('Residue Position')
        plt.ylabel('Residue Position')
    plt.savefig(f'plot/{name}_attention_contact_bw.png', dpi=600)

def plot_heatmap(probs_matrix, sequence, tokens, name=''):
    #Plots a heatmap for the given probability matrix and sequence.
    plt.figure(figsize=(50,25))
    ax = sns.heatmap(probs_matrix, xticklabels=range(1, len(sequence)+1), yticklabels=tokens, cmap="viridis")

    # Increase font size of the labels
    ax.set_yticklabels(ax.get_yticklabels(), size=15)
    
    plt.xlabel('Position in Sequence', fontsize=20)
    plt.ylabel('Amino Acid', fontsize=20)
    plt.title('Softmax Probabilities for Each Amino Acid in the Sequence', fontsize=20)

    # Rotate x-ticks
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Save the figure
    plt.savefig(f'plot/{name}_heatmap.png', dpi=600, bbox_inches='tight')

def plot_entropy(entropy_values, name=''):
    """
    Plots the entropy for each position.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(entropy_values) + 1), entropy_values)
    plt.xlabel('Position in Sequence')
    plt.ylabel('Entropy')
    plt.title('Entropy for Each Position in the Sequence')
    plt.grid(True)
    plt.savefig(f'plot/{name}_entropy_plot.png', dpi=600)

def compute_entropy(probs_matrix):
    """
    Computes the entropy for each position in the given probability matrix.
    """
    # Initialize an empty list to store the entropy values
    entropy_values = []

    # Iterate over the columns of probs_matrix
    for i in range(probs_matrix.shape[1]):
        # Compute the entropy for the probabilities at the current position
        H = entropy(probs_matrix[:, i], base=2)
        entropy_values.append(H)

    # Convert entropy_values to a numpy array for convenience
    entropy_values = np.array(entropy_values)

    return entropy_values

def find_and_mask_difference(wild_type_sequence, sequence):
    diff_position = None
    substituted_amino_acid = None
    masked_sequence = sequence  # Initialize masked sequence with original sequence

    for i in range(len(wild_type_sequence)):
        if wild_type_sequence[i] != sequence[i]:
            diff_position = i
            substituted_amino_acid = sequence[i]
            # Prepare the masked sequence
            masked_sequence = sequence[:i] + '<mask>' + sequence[i+1:]
            break

    return masked_sequence, diff_position, substituted_amino_acid

def extract_predict_masked_positions_features(model, batch_converter, wild_type_sequence, df, seq_column):
    predictions = []
    entropy_list = []
    attention_contacts = []

    for i, sequence in enumerate(df[seq_column]):
        masked_sequence, mask_position, substituted_amino_acid = find_and_mask_difference(wild_type_sequence, sequence)
        #print("masked_sequence", masked_sequence)
        #print("mask_position", mask_position)
        #print("substituted_amino_acid", substituted_amino_acid)

        if substituted_amino_acid is not None:
            data = [("protein" + str(i + 1), masked_sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, return_contacts=True)

            logits = results["logits"].squeeze(0)
            probabilities = torch.softmax(logits, dim=-1)

            substituted_amino_acid_index = batch_converter.alphabet.tok_to_idx.get(substituted_amino_acid)

            if substituted_amino_acid_index is not None:
                substituted_amino_acid_probability = probabilities[mask_position, substituted_amino_acid_index].item()
                #print("probabilities", substituted_amino_acid_probability)

                predictions.append(substituted_amino_acid_probability)

                specific_contacts = results["contacts"][0][substituted_amino_acid_index, mask_position].tolist()
                #print("contacts", specific_contacts)
                attention_contacts.append(specific_contacts)

                specific_probabilities = probabilities[mask_position].cpu()
                specific_entropy = entropy(specific_probabilities)
                #print("entropy", specific_entropy)
                entropy_list.append(specific_entropy)

    return predictions, entropy_list, attention_contacts

def convert_to_logarithmic(probabilities, entropies, attention_contacts):
    df = pd.DataFrame({'probability': probabilities})
    log_probabilities = np.log10(probabilities)
    scaled_logarithmic = -20 * (log_probabilities - np.log10(1e-5)) / (np.log10(1) - np.log10(1e-5))
    df['scaled_logarithmic'] = scaled_logarithmic
    df['mean_entropy'] = entropies
    df['contacts'] = attention_contacts
    return df

def extract_features(model, batch_converter, device, df, seq_column):
    
    # Prepare to store the results
    all_sequence_representations = []
    
    # Iterate over the DataFrame
    for i, sequence in enumerate(df[seq_column]):
        
        # Prepare data
        data = [("protein" + str(i), sequence)]
    
        # Convert batch of sequences into Torch tensors using batch_converter
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        # Extract representations
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])
            token_representations = results["representations"][33].detach().cpu().numpy()

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        for i, (_, seq) in enumerate(data):
            all_sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))


        # Clean up
        del batch_tokens, results
        gc.collect()
        torch.cuda.empty_cache()

    return all_sequence_representations

def convert_to_dense_columns(features_array):    
    df = pd.DataFrame(features_array)
    df.columns = ['Feature_' + str(x) for x in df.columns]
    return df

def plot_sequence(name, seq):
    # plot protein sequence scores using ESM
    os.makedirs('plot', exist_ok=True)
    data = [("protein0", seq)] 
    tokens = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    attention_contacts_list, sequence_representations, probs_matrix = extract_wt_features(model, batch_converter, device, data, tokens)
    plot_attention_contacts(attention_contacts_list, data, name=name)
    plot_attention_contacts_bw(attention_contacts_list, data, name=name)
    plot_heatmap(probs_matrix, seq, tokens, name=name)
    entropy_values = compute_entropy(probs_matrix)
    plot_entropy(entropy_values,name=name)

def regression_evaluation(true_value: np.ndarray, pred_value: np.ndarray) -> pd.DataFrame:
    """ ML metrics for regression models
    :param true_value
    :param pred_value
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    r2 = r2_score(true_value, pred_value)
    mae = mean_absolute_error(true_value, pred_value)
    rmse = np.sqrt(mean_squared_error(true_value, pred_value))
    mse = mean_squared_error(true_value, pred_value)
    smape = 100 / len(true_value) * np.sum(
        2 * np.abs(pred_value - true_value) / (np.abs(true_value) + np.abs(pred_value)))
    # MAPE metrics
    # from sklearn.metrics import mean_absolute_percentage_error
    # mape = mean_absolute_percentage_error(true_value, pred_value)

    dataframe = pd.DataFrame([r2, mae, rmse, mse, smape]).T
    dataframe.columns = ['r2', 'MAE', 'RMSE', 'MSE', "SMAPE"]
    return dataframe

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="ESM few-shot mutation effect prediction.")
    parser.add_argument('--name', default='ESM_mut', type=str, help='Project name for saving results.')
    parser.add_argument('--train', default='train.csv', type=str, help='Train csv.')
    parser.add_argument('--test', default='test.csv', type=str, help='Test csv.')
    parser.add_argument("--sequence", default='MQRAVSVVARLGFRLQAFPPALCRPLSCAQEVLRRTPLYDFHLAHGGKMVAFAGWSLPVQYRDSHTDSHLHTRQHCSLFDVSHMLQTKILGSDRVKLMESLVVGDIAELRPNQGTLSLFTNEAGGILDDLIVTNTSEGHLYVVSNAGCWEKDLALMQDKVRELQNQGRDVGLEVLDNALLALQGPTAAQVLQAGVADDLRKLPFMTSAVMEVFGVSGCRVTRCGYTGEDGVEISVPVAGAVHLATAILKNPEVKLAGLAARDSLRLEAGLCLYGNDIDEHTTPVEGSLSWTLGKRRRAAMDFPGAKVIVPQLKGRVQRRRVGLMCEGAPMRAHSPILNMEGTKIGTVTSGCPSPSLKKNVAMGYVPCEYSRPGTMLLVEVRRKQQMAVVSKMPFVPTNYYTLK', type=str, help="Base sequence to which mutations were applied.")
    parser.add_argument('--output_file', default='result.csv', type=str, help='Output csv file.')
    args = parser.parse_args()
    name = args.name

    # Read datasets
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    sub_df = test_df[['mutations']]

    # Split dataset into train and valid set (8:2)
    train_df, val_df = train_test_split(train_df, test_size=0.20, random_state=42)

    # Load ESM model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    name = args.name
    wt = args.sequence
    plot_sequence(name, seq=wt) # plot wild-type protein sequence scores using ESM

    train_probabilities, train_entropy, train_attention_contacts_list= extract_predict_masked_positions_features(model, batch_converter, wt, train_df, 'protein_sequence')
    valid_probabilities, valid_entropy, valid_attention_contacts_list = extract_predict_masked_positions_features(model, batch_converter, wt, val_df, 'protein_sequence')
    test_probabilities, test_entropy, test_attention_contacts_list = extract_predict_masked_positions_features(model, batch_converter, wt, test_df, 'protein_sequence')
    train_feats_df = convert_to_logarithmic(train_probabilities, train_entropy, train_attention_contacts_list)
    val_feats_df = convert_to_logarithmic(valid_probabilities, valid_entropy, valid_attention_contacts_list)
    test_feats_df = convert_to_logarithmic(test_probabilities, test_entropy, test_attention_contacts_list)
    print(train_feats_df.head(10))

    train_features = extract_features(model, batch_converter, device, train_df, 'protein_sequence')
    valid_features = extract_features(model, batch_converter, device, val_df, 'protein_sequence')
    test_features = extract_features(model, batch_converter, device, test_df, 'protein_sequence')

    train_sequence_feats_df = convert_to_dense_columns(train_features)
    val_sequence_feats_df = convert_to_dense_columns(valid_features)
    test_sequence_feats_df = convert_to_dense_columns(test_features)

    train_df["protein_length"] = train_df["protein_sequence"].apply(lambda x: len(x))
    val_df["protein_length"] = val_df["protein_sequence"].apply(lambda x: len(x))
    test_df["protein_length"] = test_df["protein_sequence"].apply(lambda x: len(x))

    train_df.drop(columns=["protein_sequence","mutations"], inplace=True)
    val_df.drop(columns=["protein_sequence","mutations"], inplace=True)
    test_df.drop(columns=["protein_sequence","mutations"], inplace=True)

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_df_concat = pd.concat([train_df, train_feats_df,train_sequence_feats_df], axis=1)
    val_df_concat = pd.concat([val_df, val_feats_df, val_sequence_feats_df], axis=1)
    test_df_concat = pd.concat([test_df, test_feats_df, test_sequence_feats_df], axis=1)

    X_train = train_df_concat.drop(columns=["label"])
    y_train = train_df_concat["label"]

    X_val = val_df_concat.drop(columns=["label"])
    y_val = val_df_concat["label"]

    X_test = test_df_concat

    model = XGBRegressor(learning_rate=0.1, max_depth=8, n_estimators=150,random_state=99)
    model.fit(X_train, y_train)

    # Checking the performance of the model on Train, Val and Test Set
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    print(X_test.shape, X_train.shape, y_train.shape)
    
    train_metrics_df = regression_evaluation(y_train, y_pred_train)
    print("Performance on Train data:")
    print(train_metrics_df)
    val_metrics_df = regression_evaluation(y_val, y_pred_val)
    print("Performance on Valid data:")
    print(val_metrics_df)
    print("Training Correlation Value: {}".format(spearmanr(y_pred_train, y_train)))
    print("Validation Correlation Value: {}".format(spearmanr(y_pred_val, y_val)))

    # Predict testing dataset
    y_pred_test = model.predict(X_test)
    sub_df.loc[:, ["predict_label"]] = y_pred_test
    sub_df.to_csv(args.output_file, index=False, sep='\t')
    print(sub_df.head(5))