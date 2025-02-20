# AI蛋白质工程

> 使用蛋白质序列概率模型推断突变效应

## AI蛋白质工程(一) —— 背景介绍与环境搭建

### 研究背景

蛋白质工程(Protein Engineering)在在生物技术、医学和工业生产等多个领域都起着至关重要的作用 [1]。蛋白质工程的目的是通过改造蛋白质序列以增强其的功能，如热稳定性和酶活性。传统方法大多基于高通量的实验方法来筛选突变。例如，定向进化通过随机突变和多轮筛选，一般来说都需要对成百上千个蛋白变体进行表达纯化和功能验证 [2]。近年来，随着人工智能(Artificial Intelligence, AI)技术的发展，计算生物学家采用计算方法来辅助蛋白质工程中的变体筛选 [3]。其中最具代表性的就是蛋白质序列概率模型(Protein Sequence Likelihood Models, PSLMs)。这种模型以其不需要功能标签(fitness label)数据，备受计算生物学家们的欢迎。

蛋白质序列概率模型是一类基于自监督深度学习算法的AI技术 [3]。通过在大规模的蛋白质数据(序列或结构数据)上进行训练，蛋白质序列概率模型可以学习天然的氨基酸的概率分布，并将学到的模型表示应用在各类下游任务中，如蛋白质设计 [4]、蛋白质工程 [5] 和蛋白质功能预测 [6]。近年来，PSLMs在蛋白质工程取得了显著的成果，改进优化了多种重要的生物酶分子，包括水解酶 [7]、抗体 [8, 9]、核酸酶 [10]、聚合酶 [10]等等。

### 蛋白质语言模型

蛋白质语言模型(Protein Language Models, PLMs)是一类在海量蛋白质序列上训练的概率模型。尽管只在蛋白质序列上训练，PLMs通过学习序列的上下文关系，已经在多个结构相关的任务上具有很好的鲁棒性，如接触预测。此外，大量的蛋白质序列中包含了非常重要的共进化信息。对于一些非常重要的氨基酸位点，自然界中很少发生突变；如果这些关键位点发生了突变，那么与之在空间中相邻的氨基酸就必须也发生突变，来适应关键位点变化对结构所带来的破坏。这种序列上的约束就是残基共进化。通过在大量的序列数据上进行训练，语言模型可以捕获道这些潜在的约束，进而训练出与蛋白质结构和功能相关的有意义的表示。

![image-20250117150117556](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117174101063.png)

蛋白质语言模型的训练目标是基于训练数据去预测一个序列中每个位置出现的可能性。给定一个序列$S=(S_1,S_2,\ ...\ ,S_N)$，假设其中第$i$个氨基酸是$S_i$，则一个蛋白质序列的概率可以近似为其所有氨基酸的联合概率。

$$
P(S)=\prod_i^NP(S_i)
$$

为了将氨基酸的顺序也进行建模，我们一般使用上文出现的所有氨基酸的联合概率来表示下一个氨基酸的概率。

$$
P(S)=\prod_i^NP(S_i\vert S_{i-(n-1),\dots,S_{i-1})})
$$

为了捕获长序列之间的依赖，最新的语言模型一般采用神经网络的架构（Transformer）来对任意长度的序列进行建模。Transformer模型最早是用在自然语言处理的任务中（如机器翻译），它包括一个编码器和一个解码器。具体来说，Transformer的模型工程过程如下：

1. **嵌入层**：首先，输入的氨基酸序列会通过一个嵌入层生成序列的初始表示，将每个氨基酸映射为一个高维向量。这些向量能够有效地表示氨基酸在序列中的语义信息。
2. **注意力层**：接下来，这些表示通过自注意力机制来捕获序列中各个氨基酸之间的长距离依赖关系。自注意力机制通过计算每个氨基酸之间的相关性来调整它们的表示，使得模型能够关注序列中不同部分的重要信息。
3. **前馈神经网络**：在每个注意力层之后，表示会通过前馈神经网络进行处理，以增强模型的表达能力。这个过程进一步提升了模型在捕捉复杂依赖关系方面的能力。
4. **输出层**：最后，经过多层处理后的表示被送入输出层，用于预测每个氨基酸的概率分布。这一层的输出即为每个位置上可能的氨基酸及其对应的概率。
5. **损失函数与优化**：在训练过程中，通过将模型输出的概率与真实的天然氨基酸进行比较，计算损失值。然后，通过优化算法（如梯度下降）来更新模型的参数，以最小化损失，从而使模型能够更好地拟合训练数据。

### ESM系列模型介绍

理解蛋白质对于许多科学研究至关重要，从药物发现到合成生物学都离不开它。传统的分析方法，如X射线晶体学和核磁共振，不仅成本高昂，而且耗费大量人力物力。因此，迫切需要能够大规模处理蛋白质分析的计算方法。随着来自UniProt和NCBI GenBank等数据库的蛋白质序列数据日益增多，开发高效的计算工具以实现蛋白质的高效分析变得尤为重要。传统的序列比对方法（如BLAST）难以应对现代蛋白质数据集的巨大多样性和复杂性。考虑到蛋白质序列与自然语言之间的相似性，计算生物学家们开始将自然语言处理（Natural Language Processing，NLP）技术应用于生物数据：将氨基酸作为“词”，将蛋白质序列作为“句子”，从而解码蛋白质的“语义”[12]。

ESM（Evolutionary Scale Modeling）系列蛋白质语言模型在大规模的蛋白质序列数据进行预训练[13-17]，可以捕获到蛋白质序列中复杂的进化和结构信息，从而在多个下游生物学任务中取得了出色的表现。

ESM系列模型包括①最早的是ESM1系列（ESM-MSA-1b、ESM1b、ESM1v）；②ESM2语言模型实现了高精度的蛋白质结构预测；③多模态的ESM3模型实现了功能蛋白的从头设计。

1. ESM-1b[13]采用Transformer架构在UniRef50数据集上进行训练，使用masked token prediction任务，即随机mask序列中的部分氨基酸（例如，15%的氨基酸），并基于序列中其它未被mask的氨基酸预测mask部分氨基酸类型。
2. ESM-MSA-1b[14]在ESM-1b的基础上进行了改进，将输入从蛋白质序列改为多序列比对（Multiple Sequence Alignment，MSA），并在Transformer中加入行、列两个轴向的注意力机制。
3. ESM-1v[15]模型使用与ESM-1b相同的架构，在数据量更大的UniRef90数据集上进行训练，能够实现蛋白质功能的零样本预测。
4. ESM2[16]模型在ESM1基础上，使用了更深层的神经网络架构和更多的训练数据，在多个下游任务中的取得了较ESM-1b更好的性能。ESM2开放了从8M到15B参数量的不同模型，供用户按需选择不同大小的模型。
5. ESM3[17]采用生成式掩码语言模型（Masked Language Model, MLM）进行训练，在训练过程中，掩码标记会以噪声调度的方式进行采样，确保ESM3能够在不同的掩码组合下进行训练。这种方式区别于传统的掩码语言建模，允许模型从任何起点生成任意顺序的标记。ESM3能够同时利用序列、结构和功能三个模态的7种信息，并且允许利用任意条件（如几个残基、结构、二级结构和功能等信息）生成目标蛋白质。

|            特征            |     ESM1     |      ESM2      |      ESM3      |
| :------------------------: | :-----------: | :-------------: | :-------------: |
|    **训练集大小**    | 小 (~1M 序列) | 中等(~10M 序列) | 大量 (~1B 序列) |
|     **模型深度**     |     浅层     |      深层      |     超级深     |
| **是否使用结构数据** |      无      |    有限结构    |    大量结构    |
|   **下游任务表现**   |     基准     |      改进      |      最优      |

### 1. 环境配置

#### 1. 下载模型权重

ESM2 提供了多个不同参数量的模型，本文选择了中等规模的 650M 模型（esm2_t33_650M_UR50D）。该模型的 HuggingFace 下载链接如下：https://huggingface.co/facebook/esm2_t33_650M_UR50D

#### 2. 创建环境

本文在 Linux Ubuntu 22.04.1 LTS 环境下进行调试，Mac OS 和 Windows 系统用户可以参考，但具体问题可能因操作系统差异而需要做相应调整。

- 安装Anaconda

  ```sh
  wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh

  bash Anaconda3-2020.07-Linux-x86_64.sh
  ```
- 创建conda环境

  ```sh
  conda create -n mutation python=3.11
  conda activate mutation
  # 安装数据分析相关包
  pip install tqdm numpy pandas matplotlib biopython
  # 安装机器学习相关包
  pip install scikit-learn xgboost scipy
  # 安装语言模型相关包
  pip install torch fair-esm transformers 
  ```
- 下载本文的python代码

  ```sh
  git clone https://github.com/Gabriel-QIN/AI_Protein_Engneering.git
  ```

### 2. 零样本突变预测

ESM预测突变影响的方式是无监督的，即在训练过程中并未使用与突变影响相关的标签进行有监督学习。

1. 首先，我们需要加载预训练的ESM-2模型及其分词器，并指定输入的蛋白质序列。接着，使用分词器对序列进行分词处理，生成一系列的token ID。在蛋白质序列中，每个氨基酸都通过分词器的词汇表映射到一个相应的token。

   ```python
   from transformers import EsmTokenizer, EsmForMaskedLM

   model = EsmForMaskedLM.from_pretrained(model_name)
   for name, param in model.named_parameters():
       if 'contact_head.regression' in name:
       param.requires_grad = False
   tokenizer = EsmTokenizer.from_pretrained(model_name)
   model = model.to(device)
   model.eval()
   vocab = tokenizer.get_vocab()
   ```
2. 对于蛋白质序列中的每个位置$i$，我们需要计算每个标准20种氨基酸的对数似然比来表示氨基酸替换对序列的影响。在每个位置$i$，目标氨基酸被mask，然后使用模型预测该位置氨基酸标记的概率分布。模型输出的每个氨基酸标记的logits通过softmax函数转化为概率。

   ```python
   import torch
   with torch.no_grad():
   	token_probs = torch.log_softmax(model(batch_tokens).logits, dim=-1)
   ```

在ESM-1v文章中[15]，作者比较了多种LLR打分方式，其中**Masked marginal**表现最优，因此本文默认采用**Masked marginal**方法进行打分。

假设$x_{mut}$和$x_{wt}$分别为突变序列和野生型序列，$x^{i}$为在第$i$个氨基酸引入mask的序列，$M$是一个突变的集合，如果序列在第1个氨基酸和第5个氨基酸发生突变，则$M=\{1,5\}$。不同的突变打分策略计算方式如下：

- **Masked marginal**：需要使用WT序列作为输入进行$L$（即序列长度）次正向传播。在突变位置引入mask标记，通过计算突变相对于野生型（Wild-Type，WT）氨基酸的概率来评估突变的得分。

  $$
  score = \sum_{i\in M}log\ p(x^i=x^i_{mut}|x-_{M})- log\ p(x^i=x^i_{wt}|x-_{M}))
  $$
- **Wildtype marginal**: 计算效率最快的打分策略，仅使用WT序列作为输入进行一次正向传播。对于位置$M$上的一组突变，得分为：

$$
score = \sum_{i\in M}log\ p(x^i=x^i_{mut}|x_{wt})- log\ p(x^i=x^i_{wt}|x_{wt}))
$$

- **Mutant marginal**: 与Wildtype marginal策略类似，只不过这里我们使用突变的序列代替。

$$
score = \sum_{i\in M}log\ p(x^i=x^i_{mut}|x_{wt})- log\ p(x^i=x^i_{wt}|x_{wt}))
$$

- **Pseudolikelihood**：使用MLM对序列进行打分。

$$
score = \sum_{i}log\ p(x^i=x^i_{mut}|x^i_{mut})- log\ p(x^i=x^i_{wt}|x^i_{wt}))
$$

#### 单点突变

##### 单点突变打分计算函数

```python
def label_row(row, sequence, token_probs, tokenizer, offset_idx):
    # 提取WT氨基酸、位置索引和突变氨基酸
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
    # 对WT序列和突变序列进行编码
    wt_encoded, mt_encoded = tokenizer.encode(wt, add_special_tokens=False)[0], tokenizer.encode(mt, add_special_tokens=False)[0]
    # 计算对数似然比
    score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
    return score.item()

sequence = "XXXXXXXXXXX"
df["esm2"] = df.apply(lambda row: label_row(row["mutant"], sequence, token_probs, tokenizer, offset_idx=1),axis=1)
```

##### 模型推理

使用Python脚本 `zero_shot_single.py`进行打分，生成所有可能的单点突变的打分，以及Top 20突变的csv文件。

该脚本的帮助文档如下：

![b22c7614b2385baf700ac9135abf1fc](https://raw.githubusercontent.com/Gabriel-QIN/source/master/pic/b22c7614b2385baf700ac9135abf1fc-17400368618503.png)

```sh
myseq=MQRAVSVVARLGFRLQAFPPALCRPLSCAQEVLRRTPLY
python zero_shot_single.py --name example  -s ${myseq} -o output/single -tk 20
```

![image-20250117165949171](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117174056259.png)

![image-20250117170013475](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117174054189.png)

| mutant | positions | esm2_t33_650M_UR50D |
| ------ | --------- | ------------------- |
| C23A   | 23        | 4.591466903686523   |
| C28L   | 28        | 4.565317153930664   |
| Q16R   | 16        | 4.4299421310424805  |
| Y39P   | 39        | 4.408666610717773   |
| E31A   | 31        | 4.338397979736328   |
| F13L   | 13        | 4.296125411987305   |
| T36R   | 36        | 4.115818977355957   |
| P37R   | 37        | 3.751159429550171   |
| R24L   | 24        | 3.5591354370117188  |
| S6A    | 6         | 3.5141654014587402  |
| A17L   | 17        | 3.498432159423828   |
| V5R    | 5         | 3.450638771057129   |
| P19A   | 19        | 3.4237990379333496  |
| V8S    | 8         | 3.3827481269836426  |

##### 绘制热图

```sh
python heatmap.py --name example -i output/single/fitness_landscape_esm2_example.csv
```

![image-20250117174956421](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117174958422.png)

#### 多点突变

```sh
myseq=MQRAVSVVARLGFRLQAFPPALCRPLSCAQEVLRRTPLY
python zero_shot_multi.py --name example  --seq ${myseq} --mut_pool output/single/top20_example.csv --max_comb 3
```

![](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117174353251.png)

这一步生成了所有3个以内的多点组合打分

![image-20250117174246166](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117174258516.png)

## 参考文献

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

[13] A. Rives *et al.*  Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences, *Proc. Natl. Acad. Sci. U.S.A.* 118:15 (2021). https://doi.org/10.1073/pnas.2016239118

[14] R. Rao *et al.*  Proceedings of the 38th International Conference on Machine Learning, *PMLR* 139:8844-8856 (2021). https://doi.org/10.1101/2021.02.12.430858

[15] J. Meier *et al.* Language models enable zero-shot prediction of the effects of mutations on protein function. In Proceedings of the 35th International Conference on Neural Information Processing Systems (2021).  https://doi.org/10.1101/2021.07.09.450648

[16] Lin, Z. *et al*. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* **379**, 1123–1130 (2023). https://doi.org/10.1126/science.ade2574

[17] T. Hayes *et al.* Simulating 500 million years of evolution with a language model. *Science*, eads0018 (2024). https://doi.org/10.1126/science.ads0018

[18] https://esm3academy.com/the-evolution-of-esm-models-leading-to-esm3/

[19] https://github.com/facebookresearch/esm

[20] https://github.com/amelie-iska/Variant-Effects

[21] https://github.com/ntranoslab/esm-variants
