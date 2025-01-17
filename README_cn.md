# AI蛋白质工程

> **使用蛋白质序列概率模型推断突变效应**

## AI蛋白质工程(一) —— 背景介绍与环境搭建

### 研究背景

蛋白质工程(Protein Engineerings)在在生物技术、医学和工业生产等多个领域都起着至关重要的作用 [1]。蛋白质工程的目的是通过改造蛋白质序列以增强其的功能，如热稳定性和酶活性。传统方法大多基于高通量的实验方法来筛选突变。例如，定向进化通过随机突变和多轮筛选，一般来说都需要对成百上千个蛋白变体进行表达纯化和功能验证 [2]。近年来，随着人工智能(Artificial Intelligence, AI)技术的发展，计算生物学家采用计算方法来辅助蛋白质工程中的变体筛选 [3]。其中最具代表性的就是蛋白质序列概率模型(Protein Sequence Likelihood Models, PSLMs)。这种模型以其不需要功能标签(fitness label)数据，备受计算生物学家们的欢迎。

蛋白质序列概率模型是一类基于自监督深度学习算法的AI技术 [3]。通过在大规模的蛋白质数据(序列或结构数据)上进行训练，蛋白质序列概率模型可以学习天然的氨基酸的概率分布，并将学到的模型表示应用在各类下游任务中，如蛋白质设计 [4]、蛋白质工程 [5] 和蛋白质功能预测 [6]。近年来，PSLMs在蛋白质工程取得了显著的成果，改进优化了多种重要的生物酶分子，包括水解酶 [7]、抗体 [8, 9]、核酸酶 [10]、聚合酶 [10]等等。

### 蛋白质语言模型

蛋白质语言模型(Protein Language Models, PLMs)是一类在海量蛋白质序列上训练的概率模型。尽管只在蛋白质序列上训练，PLMs通过学习序列的上下文关系，已经在多个结构相关的任务上具有很好的鲁棒性，如接触预测。此外，大量的蛋白质序列中包含了非常重要的共进化信息。对于一些非常重要的氨基酸位点，自然界中很少发生突变；如果这些关键位点发生了突变，那么与之在空间中相邻的氨基酸就必须也发生突变，来适应关键位点变化对结构所带来的破坏。这种序列上的约束就是残基共进化。通过在大量的序列数据上进行训练，语言模型可以捕获道这些潜在的约束，进而训练出与蛋白质结构和功能相关的有意义的表示。

![image-20250117150117556](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117174101063.png)

蛋白质语言模型的训练目标是基于训练数据去预测一个序列中每个位置出现的可能性。给定一个序列 **$S=(S_1,S_2, ...\, S_N)$**，假设其中第**i**个氨基酸是**S_i**，则一个蛋白质序列的概率可以近似为其所有氨基酸的联合概率

$$
P(S)=\prod_i^NP(S_i)
$$

为了将氨基酸的顺序也进行建模，我们一般使用上文出现的所有氨基酸的联合概率来表示下一个氨基酸的概率。

$$
P(S)=\prod_i^NP(S_i\vert S_{i-(n-1),\dots,S_{i-1})})
$$

为了捕获长序列之间的依赖，最新的语言模型一般采用神经网络的架构（Transformer）来对任意长度的序列进行建模。Transformer模型最早是用在自然语言处理的任务中（如机器翻译），它包括一个编码器和一个解码器。具体来说，Transformer的模型工程过程如下：

1. **嵌入层**：首先，输入的基酸序列会通过一个嵌入层生成序列的初始表示，将每个氨基酸映射为一个高维向量。这些向量能够有效地表示氨基酸在序列中的语义信息。
2. **注意力层**：接下来，这些表示通过自注意力机制来捕获序列中各个氨基酸之间的长距离依赖关系。自注意力机制通过计算每个氨基酸之间的相关性来调整它们的表示，使得模型能够关注序列中不同部分的重要信息。
3. **前馈神经网络**：在每个注意力层之后，表示会通过前馈神经网络进行处理，以增强模型的表达能力。这个过程进一步提升了模型在捕捉复杂依赖关系方面的能力。
4. **输出层**：最后，经过多层处理后的表示被送入输出层，用于预测每个氨基酸的概率分布。这一层的输出即为每个位置上可能的氨基酸及其对应的概率。
5. **损失函数与优化**：在训练过程中，通过将模型输出的概率与真实的天然氨基酸进行比较，计算损失值。然后，通过优化算法（如梯度下降）来更新模型的参数，以最小化损失，从而使模型能够更好地拟合训练数据。

### ESM蛋白质语言模型

至今，计算生物学家开发了许多的蛋白质语言模型 [12]，如ProtTrans系列 [13]，ESM系列 [14]，ProST [15] 和SaProt [16]。为了使用语言模型进行突变效应预测，本文采用了ESM2 [17] 的模型表示进行零样本预测。

### 1. 环境配置

#### 1. 下载模型权重

ESM2 提供了多个不同参数量的模型，本文选择了中等规模的 650M 模型（esm2_t33_650M_UR50D）。该模型的 HuggingFace 下载链接如下：[https://huggingface.co/facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D)

#### 2. 创建环境

本文在 Linux Ubuntu 22.04.1 LTS 环境下进行调试，Mac OS 和 Windows 系统用户可以参考，但具体问题可能因操作系统差异而需要做相应调整。

* **安装Anaconda**

  ```
  wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh

  bash Anaconda3-2020.07-Linux-x86_64.sh
  ```
* **创建conda环境**

  ```
  conda create -n mutation python=3.11
  conda activate mutation
  # 安装数据分析相关包
  pip install tqdm numpy pandas matplotlib biopython
  # 安装机器学习相关包
  pip install scikit-learn xgboost scipy
  # 安装语言模型相关包
  pip install torch fair-esm transformers 
  ```

### 2. 零样本突变预测

#### 单点突变

##### 模型推理

```
myseq=MQRAVSVVARLGFRLQAFPPALCRPLSCAQEVLRRTPLY
python zero_shot_single.py --name example  -s ${myseq} -o output/single -tk 20
```

生成所有可能的单点突变的打分，以及Top 20突变的csv文件。

![image-20250117165949171](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117174056259.png)

![image-20250117170013475](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117174054189.png)

##### 绘制热图

```
python heatmap.py --name example -i output/single/fitness_landscape_esm2_example.csv
```

![image-20250117174956421](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117174958422.png)

#### 多点突变

```
myseq=MQRAVSVVARLGFRLQAFPPALCRPLSCAQEVLRRTPLY
python zero_shot_multi.py --name example  --seq ${myseq} --mut_pool output/single/top20_example.csv --max_comb 3
```

![](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117174353251.png)

这一步生成了所有3个以内的多点组合打分

![image-20250117174246166](https://raw.githubusercontent.com/Gabriel-QIN/pic/master/20250117174258516.png)

### 下期预告

少样本微调预测突变

使用逆折叠模型进行突变预测

## 参考文献

**[1] Listov, D., Goverde, C.A., Correia, B.E. ***et al.* Opportunities and challenges in design and optimization of protein function. *Nat Rev Mol Cell Biol***25**, 639–653 (2024). [https://doi.org/10.1038/s41580-024-00718-y](https://doi.org/10.1038/s41580-024-00718-y)

**[2] Romero, P., Arnold, F. Exploring protein fitness landscapes by directed evolution. ***Nat Rev Mol Cell Biol***10**, 866–876 (2009). [https://doi.org/10.1038/nrm2805](https://doi.org/10.1038/nrm2805)

**[3] Reeves, S., Kalyaanamoorthy, S. Zero-shot transfer of protein sequence likelihood models to thermostability prediction. ***Nat Mach Intell***6**, 1063–1076 (2024). [https://doi.org/10.1038/s42256-024-00887-7](https://doi.org/10.1038/s42256-024-00887-7)

**[4] Chu, A.E., Lu, T. & Huang, PS. Sparks of function by de novo protein design. ***Nat Biotechnol***42**, 203–215 (2024). [https://doi.org/10.1038/s41587-024-02133-2](https://doi.org/10.1038/s41587-024-02133-2)

**[5] Notin, P., Rollins, N., Gal, Y. ***et al.* Machine learning for functional protein design. *Nat Biotechnol***42**, 216–228 (2024). [https://doi.org/10.1038/s41587-024-02127-0](https://doi.org/10.1038/s41587-024-02127-0)

**[6] Song, Y., Yuan, Q., Chen, S. ***et al.* Accurately predicting enzyme functions through geometric graph learning on ESMFold-predicted structures. *Nat Commun***15**, 8180 (2024). [https://doi.org/10.1038/s41467-024-52533-w](https://doi.org/10.1038/s41467-024-52533-w)

**[7] Lu, H., Diaz, D.J., Czarnecki, N.J. ***et al.* Machine learning-aided engineering of hydrolases for PET depolymerization. *Nature***604**, 662–667 (2022). [https://doi.org/10.1038/s41586-022-04599-z](https://doi.org/10.1038/s41586-022-04599-z)

**[8] Hie, B.L., Shanker, V.R., Xu, D. ***et al.* Efficient evolution of human antibodies from general protein language models. *Nat Biotechnol***42**, 275–283 (2024). [https://doi.org/10.1038/s41587-023-01763-2](https://doi.org/10.1038/s41587-023-01763-2)

**[9] Varun R. Shanker ***et al.* Unsupervised evolution of protein and antibody complexes with a structure-informed language model. *Science***385**, 46-53 (2024) .[https://doi.org/10.1038/](https://doi.org/10.1038/)[10.1126/science.adk8946](https://doi.org/10.1126/science.adk8946)

**[10] Kaiyi J. ***et al*. , Rapid in silico directed evolution by a protein language model with EVOLVEpro. *Science***0**, eadr6006. [https://doi.org/10.1126/science.adr6006](https://doi.org/10.1126/science.adr6006)

**[11] Ruffolo, J.A., Madani, A. Designing proteins with language models. ***Nat Biotechnol***42**, 200–202 (2024). [https://doi.org/10.1038/s41587-024-02123-4](https://doi.org/10.1038/s41587-024-02123-4)

**[12] **[https://github.com/LirongWu/awesome-protein-representation-learning](https://github.com/LirongWu/awesome-protein-representation-learning)

**[13] **[https://github.com/agemagician/ProtTrans](https://github.com/agemagician/ProtTrans)

**[14] **[https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm)

**[15] **[https://github.com/mheinzinger/ProstT5](https://github.com/mheinzinger/ProstT5)

**[16] **[https://github.com/westlake-repl/SaProt](https://github.com/westlake-repl/SaProt)

**[17] Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. ***Science***379**, 1123–1130 (2023). [https://doi.org/10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)

**[18] **[https://github.com/amelie-iska/Variant-Effects](https://github.com/amelie-iska/Variant-Effects)

**[19] **[https://github.com/ntranoslab/esm-variants](https://github.com/ntranoslab/esm-variants)
