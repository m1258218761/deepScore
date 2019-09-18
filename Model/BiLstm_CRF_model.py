# coding=utf-8
import torch
import torch.nn as nn
from torch.nn import utils as nn_utils
import numpy as np
from torchcrf import CRF

##############################
#########  BILSTM + CRF
##############################
''' Python libraries need to be installed :  pytorch-crf'''

torch.manual_seed(1)

class BiLSTM_CRF(nn.Module):
    def __init__(self,hidden_dim,features_number,layer_num=2,batch_size=16,label_number=4):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.label_number = label_number
        self.layer_num = layer_num
        self.batch_size = batch_size
        self.features_number = features_number
        self.lstm = nn.LSTM(self.features_number, hidden_dim // 2,num_layers=self.layer_num, bidirectional=True,dropout=0.2).cuda()
        self.out_layer = nn.Linear(self.hidden_dim,self.label_number*11).cuda()
        self.crf = CRF(11,batch_first=True)

    def init_hidden(self):
        return (torch.randn(self.layer_num*2, self.batch_size, self.hidden_dim // 2).cuda(),
                torch.randn(self.layer_num*2, self.batch_size, self.hidden_dim // 2).cuda())

    def get_lstm_features(self, sentence,length):
        self.hidden = self.init_hidden()
        sorted_length,sorted_index = torch.sort(length,0,descending=True)
        _,unsorted_index = torch.sort(sorted_index,0)
        packed = nn_utils.rnn.pack_padded_sequence(torch.index_select(sentence,0,sorted_index),sorted_length,batch_first=True)
        lstm_out, self.hidden = self.lstm(packed, self.hidden)
        unpacked,unpacked_length = nn_utils.rnn.pad_packed_sequence(lstm_out,batch_first=True)
        unpacked = torch.index_select(unpacked,0,unsorted_index)        # Shape : batch, seq_len, hidden_size * num_directions
        lstm_out = self.out_layer(unpacked)
        lstm_out = lstm_out.reshape(lstm_out.shape[0],lstm_out.shape[1],self.label_number,-1)    # Shape : batch,seq_len,label_number,11
        return lstm_out

    def forward(self, x, y_true, batch_length):
        x = x.permute(0,2,1)
        print(x.shape)
        out = self.get_lstm_features(x.reshape(len(batch_length),max(batch_length),self.features_number),batch_length)
        mask = y_true != -1
        loss = self.crf(out.view(out.size(0),-1,11),y_true.view(y_true.size(0),-1),mask=mask.view(mask.size(0),-1))
        loss = -loss/self.batch_size
        #   Alternative : We modify some of the contents of the library and add the marginal probability calculation function.
        # p = self.crf.compute_log_marginal_probabilities(out.view(out.size(0),-1,11),mask=mask.view(mask.size(0),-1))
        out = self.crf.decode(out.view(out.size(0),-1,11),mask=mask.view(mask.size(0),-1))
        y_t = []
        y_p = []
        results = []
        for i in range(len(batch_length)):
            _tag = y_true[i][:batch_length[i]].view(-1)
            _tag = np.array(_tag.cpu()).tolist()
            _out = out[i]
            y_t.extend(_tag)
            y_p.extend(_out)
            results.append(_tag)
            results.append(_out)
        return y_t,y_p,results,loss,''

def BiLstm_CRF(hidden_dim,features_number,layer_num=2,batch_size=16,label_number=4):
    return BiLSTM_CRF(hidden_dim,features_number,layer_num,batch_size,label_number)

#   Ordinal Regression, just for test.
def y_true_to_ordinal(y_true):
    t_tag=[]
    for t in y_true:
        temp = [0]*10
        for i in range(t):
           temp[i]=1
        t_tag.extend(temp)
    return torch.tensor(t_tag)
def ordinal_predict(self,feats):
    p = self.sig(feats)
    i = p > 0.5
    pred_tag = torch.sum(i.view(-1,10),1)
    return pred_tag,p,i.float()

if __name__ == '__main__':
    tran_tags = y_true_to_ordinal(torch.tensor([1,2,3,10,0]))
    print(tran_tags)