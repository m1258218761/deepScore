# deepScore-α
![](https://img.shields.io/badge/language-python-blue.svg) ![](https://img.shields.io/badge/frame-pytorch-orange.svg) ![](https://img.shields.io/badge/license-MIT-000000.svg)

A Peptide Spectrum Match Scoring Algorithm based on Deep Learning Model.
This algorithm extracts candidate peptide sequences from Comet and MSGF+ for re-scoring and comparison with the original results: 
+ The number of peptide retained by deepScore-α when FDR=0.01 in human proteome dataset increased by about 14% compared with Comet and MSGF+, and the Top1 hit ratio (the proportion of spectrum with the heighest score of correct peptide sequence) increased by about 5%. 

+ The generalization performance test on ProteomeTools2 dataset which used the model trained by Humanbody dataset showed that the peptide retained by deepScore-α at FDR=0.01 improved by about 7% compared with Comet and MSGF+, the Top1 hit rate increased by about 5%, and the identification results from Decoy library in Top1 decreased by about 60%.
### Prerequisites
+ python 3.6+
+ pytorch 1.0+
### Structure diagram of deep learning model adopted by deepScore-α
> deepScore-α's model is One-dimensional [Resnet](https://arxiv.org/abs/1512.03385) and combines [Multi-Head self-Attention](https://arxiv.org/abs/1706.03762).
<img src="img/model structure.png" width="35%" high="35%"></img>
### The framework of the deepScore-α algorithm
> Model training is needed before scoring with deepScore-α,and the output of the model is different at different stages.
><img src="img/process flow.png" width="50%" high="50%"></img>

We also tried to use model BiLstm+CRF, but did't use it in the end.
The CRF need to install python libraries named [pytorch-crf](https://github.com/kmkurn/pytorch-crf).