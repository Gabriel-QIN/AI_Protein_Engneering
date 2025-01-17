# AI Protein Engineering

## Research Background

Protein engineering plays a crucial role in various fields such as biotechnology, medicine, and industrial production [1]. The goal of protein engineering is to modify protein sequences to enhance their functions, such as thermal stability and enzyme activity. Traditional methods mostly rely on high-throughput experimental approaches for screening mutations. For example, directed evolution involves random mutations and multiple rounds of screening, typically requiring the expression, purification, and functional validation of hundreds or thousands of protein variants [2]. In recent years, with the development of Artificial Intelligence (AI) technologies, computational biologists have turned to computational methods to assist in variant screening in protein engineering [3]. One of the most representative methods is Protein Sequence Likelihood Models (PSLMs). These models are popular among computational biologists due to their ability to work without functional label (fitness label) data.

Protein Sequence Likelihood Models are a class of AI techniques based on self-supervised deep learning algorithms [3]. By training on large-scale protein data (either sequence or structure data), PSLMs can learn the probability distribution of natural amino acids, and the learned model representations are applied to various downstream tasks, such as protein design [4], protein engineering [5], and protein function prediction [6]. In recent years, PSLMs have achieved remarkable results in protein engineering, improving and optimizing several important biological enzymes, including hydrolases [7], antibodies [8, 9], nucleases [10], polymerases [10], and more.

## Protein Language Models

Protein Language Models (PLMs) are a class of probabilistic models trained on vast amounts of protein sequence data. Although trained solely on protein sequences, PLMs learn the contextual relationships within these sequences and have demonstrated strong robustness in several structure-related tasks, such as contact prediction. Additionally, large protein sequence datasets contain crucial co-evolution information. For some important amino acid sites, mutations are rarely observed in nature; if these key sites mutate, amino acids adjacent to them in space must also mutate to adapt to the structural damage caused by the changes at the key sites. This constraint in the sequence is called residue co-evolution. By training on a large amount of sequence data, language models can capture these potential constraints, leading to meaningful representations related to protein structure and function.

The training objective of a protein language model is to predict the likelihood of each position in a sequence based on the training data. Given a sequence $S=(S_1,S_2,\ ...\ ,S_N)$, where the i-th amino acid is $S_i$, the probability of the protein sequence can be approximated by the joint probability of all amino acids:

$$
P(S)=\prod_i^NP(S_i)
$$

To model the sequence of amino acids, we typically use the joint probability of previous amino acids observed to represent the probability of the next amino acid:

$$
P(S)=\prod_i^NP(S_i\vert S_{i-(n-1),\dots,S_{i-1})})
$$

To capture dependencies across long sequences, modern language models typically adopt neural network architectures (e.g., Transformer) to model sequences of arbitrary length. The Transformer model was originally used for natural language processing tasks (such as machine translation) and consists of an encoder and a decoder. Specifically, the Transformer model engineering process is as follows:

1. **Embedding Layer**: First, the input amino acid sequence passes through an embedding layer to generate the initial representations of the sequence, mapping each amino acid to a high-dimensional vector. These vectors effectively represent the semantic information of the amino acids within the sequence.
2. **Attention Layer**: The representations are then processed by a self-attention mechanism to capture long-range dependencies between amino acids in the sequence. The self-attention mechanism adjusts the representations by computing the correlations between each amino acid, allowing the model to focus on important information in different parts of the sequence.
3. **Feed-Forward Neural Network**: After each attention layer, the representations are processed by a feed-forward neural network to enhance the model’s expressive power. This step further improves the model’s ability to capture complex dependencies.
4. **Output Layer**: Finally, the representations processed through multiple layers are passed into the output layer to predict the probability distribution of each amino acid. The output from this layer represents the possible amino acids at each position and their corresponding probabilities.
5. **Loss Function and Optimization**: During training, the predicted probabilities from the model are compared with the true natural amino acids, and the loss value is computed. The model parameters are then updated through optimization algorithms (e.g., gradient descent) to minimize the loss, allowing the model to better fit the training data.

![image-20250117174021209](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117174023796.png)

### ESM Protein Language Models

To date, computational biologists have developed many protein language models [12], such as the ProtTrans series [13], the ESM series [14], ProST [15], and SaProt [16]. In this work, we use the ESM2 [17] model representation for zero-shot mutation effect prediction.

#### 1. Download Model Weights

ESM2 provides several models with varying parameter sizes. In this study, we select the medium-sized 650M model (esm2_t33_650M_UR50D). The HuggingFace download link for this model is as follows: https://huggingface.co/facebook/esm2_t33_650M_UR50D

#### 2. Create Environment

This work was debugged in a Linux Ubuntu 22.04.1 LTS environment. macOS and Windows users can refer to this, but specific issues may need to be addressed based on operating system differences.

- Anaconda installation

  ```sh
  wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh

  bash Anaconda3-2020.07-Linux-x86_64.sh
  ```
- Create conda environment

  ```sh
  conda create -n mutation python=3.11
  conda activate mutation
  # data analysis packages
  pip install tqdm numpy pandas matplotlib biopython
  # ML packages
  pip install scikit-learn xgboost scipy
  # language model packages
  pip install torch fair-esm transformers 
  ```

### Zero-shot variant effect inference

#### single-site mutation

##### model inference

```sh
myseq=MQRAVSVVARLGFRLQAFPPALCRPLSCAQEVLRRTPLY
python zero_shot_single.py --name example  -s ${myseq} -o output/single -tk 20
```

This step help you generate scores for all possible single-point mutations and the top 20 mutations in a CSV file.

![image-20250117165949171](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117174038905.png)

![image-20250117170013475](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117174044420.png)

##### heatmap visualization

```sh
python heatmap.py --name example -i output/single/fitness_landscape_esm2_example.py
```

![](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117175025381.png)

#### multi-site mutation

```sh
myseq=MQRAVSVVARLGFRLQAFPPALCRPLSCAQEVLRRTPLY
python zero_shot_multi.py --name example  --seq ${myseq} --mut_pool output/single/top20_example.csv --max_comb 3
```

![image-20250117174347035](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117174348376.png)

Then, we have something like this:

![image-20250117174246166](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117174249446.png)

### Future updates

Few-shot fine-tuning for variant effect inference

Mutation effect prediction using inverse folding models (e.g., ProteinMPNN)

## Reference

[1] Listov, D., Goverde, C.A., Correia, B.E. *et al.* Opportunities and challenges in design and optimization of protein function. *Nat Rev Mol Cell Biol* **25**, 639–653 (2024). https://doi.org/10.1038/s41580-024-00718-y

[2] Romero, P., Arnold, F. Exploring protein fitness landscapes by directed evolution. *Nat Rev Mol Cell Biol* **10**, 866–876 (2009). https://doi.org/10.1038/nrm2805

[3] Reeves, S., Kalyaanamoorthy, S. Zero-shot transfer of protein sequence likelihood models to thermostability prediction. *Nat Mach Intell* **6**, 1063–1076 (2024). https://doi.org/10.1038/s42256-024-00887-7

[4] Chu, A.E., Lu, T. & Huang, PS. Sparks of function by de novo protein design. *Nat Biotechnol* **42**, 203–215 (2024). https://doi.org/10.1038/s41587-024-02133-2

[5] Notin, P., Rollins, N., Gal, Y. *et al.* Machine learning for functional protein design. *Nat Biotechnol* **42**, 216–228 (2024). https://doi.org/10.1038/s41587-024-02127-0

[6] Song, Y., Yuan, Q., Chen, S. *et al.* Accurately predicting enzyme functions through geometric graph learning on ESMFold-predicted structures. *Nat Commun* **15**, 8180 (2024). https://doi.org/10.1038/s41467-024-52533-w

[7] Lu, H., Diaz, D.J., Czarnecki, N.J. *et al.* Machine learning-aided engineering of hydrolases for PET depolymerization. *Nature* **604**, 662–667 (2022). https://doi.org/10.1038/s41586-022-04599-z

[8] Hie, B.L., Shanker, V.R., Xu, D. *et al.* Efficient evolution of human antibodies from general protein language models. *Nat Biotechnol* **42**, 275–283 (2024). https://doi.org/10.1038/s41587-023-01763-2

[9] Varun R. Shanker *et al.* Unsupervised evolution of protein and antibody complexes with a structure-informed language model. *Science* **385**, 46-53 (2024) .https://doi.org/10.1038/[10.1126/science.adk8946](https://doi.org/10.1126/science.adk8946)

[10] Kaiyi J. *et al*. , Rapid in silico directed evolution by a protein language model with EVOLVEpro. *Science* **0**, eadr6006. https://doi.org/10.1126/science.adr6006

[11] Ruffolo, J.A., Madani, A. Designing proteins with language models. *Nat Biotechnol* **42**, 200–202 (2024). https://doi.org/10.1038/s41587-024-02123-4

[12] https://github.com/LirongWu/awesome-protein-representation-learning

[13] https://github.com/agemagician/ProtTrans

[14] https://github.com/facebookresearch/esm

[15] https://github.com/mheinzinger/ProstT5

[16] https://github.com/westlake-repl/SaProt

[17] Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* **379**, 1123–1130 (2023). https://doi.org/10.1126/science.ade2574

[18] https://github.com/amelie-iska/Variant-Effects

[19] https://github.com/ntranoslab/esm-variants
