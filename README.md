# P-score
A Peptide Spectrum Match Scoring Algorithm based on Deep Learning Model.
### Prerequisites
+ python 3.6+
+ pytorch 1.0+
### Deep Learning Model Structure
> P-score's model is One-dimensional [Resnet](https://arxiv.org/abs/1512.03385) and combines [Multi-Head self-Attention](https://arxiv.org/abs/1706.03762).
<img src="img/model structure1.png" width="35%" high="35%"></img>
### P-score Process Flow
> Model training is needed before scoring with P-score,and the output of the model is different at different stages.
><img src="img/process flow1.png" width="50%" high="50%"></img>

We also tried to use model BiLstm+CRF, but did't use it in the end.
The CRF need to install python libraries named [pytorch-crf](https://github.com/kmkurn/pytorch-crf).