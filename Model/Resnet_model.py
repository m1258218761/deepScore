# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Attention import MultiHeadedAttention

##############################
#########  ResNet
##############################

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),

            # nn.ReLU(inplace=True),
            # nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm1d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock,batch_size=16,weight=[],feature_size=174):
        super(ResNet, self).__init__()
        self.inchannel = feature_size
        self.feature_size = feature_size
        self.weight = 2-torch.tensor(weight)
        self.batch_size = batch_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.feature_size, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=1)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=1)
        self.layer5 = self.make_layer(ResidualBlock, 1024, 2, stride=1)
        # self.attn1 = MultiHeadedAttention(2,64)
        # self.attn2 = MultiHeadedAttention(2,128)
        # self.attn3 = MultiHeadedAttention(2,256)
        # self.attn4 = MultiHeadedAttention(2,512)
        self.attn5 = MultiHeadedAttention(8,1024)
        self.attn_shortcut = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, stride=1, bias=False),nn.BatchNorm1d(1024))
        self.fc = nn.Linear(1024, 44)
        self.soft = nn.Softmax(dim=3)
        # self.pe = PositionalEncoding(46,0.0)
        self.loss = nn.CrossEntropyLoss(weight=self.weight,ignore_index=-1,size_average=False)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x, y_true,batch_length):
        # print('input size: ' + str(x.shape))
        # out = self.conv1(x)
        # print('after conv1: ' + str(out.shape))
        # x = self.pe(x.permute(0,2,1)).permute(0,2,1)
        out = self.layer1(x)
        # out = out.permute(0,2,1)
        # out,_ = self.attn1(out,out,out)
        # out = out.permute(0,2,1)

        # print('after layer1: ' + str(out.shape))
        out = self.layer2(out)
        # out = out.permute(0,2,1)
        # out,_ = self.attn2(out,out,out)
        # out = out.permute(0,2,1)

        # print('after layer2: ' + str(out.shape))
        out = self.layer3(out)
        # out = out.permute(0,2,1)
        # out,_ = self.attn3(out,out,out)
        # out = out.permute(0,2,1)

        # print('after layer3: ' + str(out.shape))
        out = self.layer4(out)
        # out = out.permute(0,2,1)
        # out,_ = self.attn4(out,out,out)
        # out = out.permute(0,2,1)

        # print('after layer4: ' + str(out.shape))
        out = self.layer5(out)
        out = out.permute(0,2,1)
        out,_attn = self.attn5(out,out,out)
        # print('after layer5: ' + str(out.shape))
        # out = F.max_pool2d(out, 4)
        # print('after max_pool: ' + str(out.shape))
        # att_shortcut = self.attn_shortcut(out).permute(0,2,1)
        out = self.fc(out)
        total_loss = self.loss(out.view(-1,11),y_true.view(-1))
        _loss = total_loss/self.batch_size

        out = self.soft(out.view(out.size(0),out.size(1),4,11))
        pred_label = torch.max(out,3)[1]
        pred_P = torch.max(out,3)[0]
        Matrix_P = []
        y_t = []
        y_p = []
        results = []
        Attn = []
        P = []
        _attn = _attn.squeeze()
        for n in range(len(batch_length)):
            _out = pred_label[n][:batch_length[n]].view(-1)
            _matrix = out[n][:batch_length[n]].view(-1,11)
            _P = pred_P[n][:batch_length[n]].view(-1)
            _tag = y_true[n][:batch_length[n]].view(-1)
            _out = np.array(_out.cpu()).tolist()
            _tag = np.array(_tag.cpu()).tolist()
            _P = _P.cpu().detach().numpy().tolist()
            _matrix = _matrix.cpu().detach().numpy().tolist()
            _a = _attn[n][:batch_length[n],:batch_length[n]]
            _a = np.array(_a.cpu().detach()).tolist()
            y_p.extend(_out)
            y_t.extend(_tag)
            results.append(_tag)
            results.append(_out)
            P.append(_P)
            Matrix_P.append(_matrix)
            Attn.append(_a)
        return y_t,y_p,results,_loss,[P,Matrix_P]


def ResNet18(batch_size,weight,feature_size):
    return ResNet(ResidualBlock,batch_size=batch_size,weight=weight,feature_size=feature_size)
