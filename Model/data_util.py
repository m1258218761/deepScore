# coding=utf-8
import os
import random
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler

class data(object):
    def __init__(self,workpath,label_number,features_norm=False,Standardization=False,discretization=0,run_model='Train',train_file='',validation_file='',test_file='',discretization_f = []):
        self.workpath = workpath
        self.train_file = train_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.label_number = label_number
        self.run_model = run_model
        self.Standardization = Standardization      ##是否对特征进行标准化处理，True：是，False：否
        self.discretization = discretization        ##0:等宽离散，1：等频离散，2：回归，不进行离散化
        self.discretization_f = discretization_f    ##等频离散分段阈值,内置变量，自动获取，无需手动添加
        self.features_norm = features_norm          ##碱性、质量、螺旋性、疏水性、等电点是否进行/Max了处理，True：进行了处理，False：未进行处理
        if self.features_norm:
            ## t/max
            self.dicB = {'N': 0.8978902953586498, 'Q': 0.9037974683544303, 'D': 0.880168776371308, 'Y': 0.8991561181434599, 'W': 0.9118143459915612,
                       'I': 0.889451476793249, 'L': 0.8843881856540085, 'V': 0.880590717299578, 'c': 0.8700421940928269, 'P': 0.9046413502109705,
                       'm': 0.9, 'MissV': 0.0, 'K': 0.9358649789029536, 'T': 0.8932489451476793, 'A': 0.8708860759493671, 'H': 0.9438818565400844,
                       'R': 1.0, 'C': 0.8700421940928269, 'F': 0.8949367088607595, 'G': 0.8552742616033755, 'E': 0.9097046413502109, 'n': 0.8978902953586498,
                       'S': 0.8759493670886076, 'M': 0.9}
            self.dicM = {'V': 0.5323988594046454, 'D': 0.6181608323113273, 'E': 0.6934816714418974, 'm': 0.7901759595812781, 'L': 0.6077196985352155,
                       'T': 0.5430355280815122, 'K': 0.6883890580571952, 'n': 0.6181608324456787, 'Q': 0.6881935231564403, 'Y': 0.8763108933017181,
                       'A': 0.38175718114350515, 'C': 0.5535767697078718, 'S': 0.4677146889509421, 'R': 0.8388955681494804, 'N': 0.6128726786518177,
                       'MissV': 0.0, 'I': 0.6077196985352155, 'H': 0.7365617907241521, 'F': 0.7903533801202285, 'M': 0.7042184479690119, 'c': 0.8600131099989604,
                       'G': 0.30643634201293507, 'W': 1.0, 'P': 0.521566650452971}
            self.dicS = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
                     'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
                     'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
                     'c': 20, 'm': 21, 'n': 22, 'MissV': -1}     #氨基酸编号
            self.dicHe = {'c': 0.6124031007751938, 'K': 0.6821705426356589, 'H': 0.751937984496124, 'P': 0.441860465116279, 'I': 1.0, 'T': 0.8449612403100776,
                        'n': 0.7286821705426356, 'm': 0.945736434108527, 'Q': 0.7441860465116279, 'R': 0.7364341085271318, 'Y': 0.8604651162790699,
                        'G': 0.8914728682170542, 'A': 0.9612403100775193, 'N': 0.7286821705426356, 'M': 0.945736434108527, 'F': 0.9767441860465116,
                        'E': 0.6589147286821705, 'V': 0.9844961240310077, 'S': 0.7751937984496123, 'W': 0.8294573643410853, 'L': 0.9922480620155039,
                        'MissV': 0.0, 'C': 0.6124031007751938, 'D': 0.689922480620155}
            self.dicHy = {'M': 0.646, 'A': 0.032, 'L': 0.952, 'K': -1.0, 'P': -0.984, 'C': 0.5, 'E': -0.3, 'S': -0.5700000000000001, 'I': 0.882,
                        'H': -0.9259999999999999, 'Y': 0.4, 'V': 0.604, 'G': -0.662, 'MissV': 0.0, 'T': -0.21600000000000003, 'm': 0.646, 'R': -0.554,
                        'N': -0.758, 'c': 0.5, 'W': 0.976, 'F': 1.0, 'Q': -0.5519999999999999, 'n': -0.758, 'D': -0.49800000000000005}
            self.dicP = {'H': 0.7053903345724907, 'c': 0.466542750929368, 'T': 0.6068773234200744, 'N': 0.5027881040892194, 'V': 0.5548327137546468,
                       'n': 0.5027881040892194, 'R': 1.0, 'S': 0.5278810408921933, 'Q': 0.525092936802974, 'K': 0.9052044609665428, 'M': 0.5343866171003717,
                       'E': 0.2992565055762082, 'G': 0.5548327137546468, 'Y': 0.5260223048327137, 'P': 0.5855018587360594, 'MissV': 0.0, 'D': 0.2760223048327138,
                       'I': 0.5594795539033457, 'L': 0.5557620817843867, 'C': 0.466542750929368, 'W': 0.5473977695167286, 'm': 0.5343866171003717,
                       'F': 0.5092936802973979, 'A': 0.5594795539033457}
            ## t/max
            self.PROTON = 0.005413156628447999
            self.H = 0.0054161046488816296
            self.O = 0.08595751115063499
            self.H2O = self.H * 2 + self.O
        else:
            self.dicB = {'A': 206.4, 'C': 206.2, 'D': 208.6, 'E': 215.6, 'F': 212.1,
                     'G': 202.7, 'H': 223.7, 'I': 210.8, 'K': 221.8, 'L': 209.6,
                     'M': 213.3, 'N': 212.8, 'P': 214.4, 'Q': 214.2, 'R': 237.0,
                     'S': 207.6, 'T': 211.7, 'V': 208.7, 'W': 216.1, 'Y': 213.1,
                     'c': 206.2, 'm': 213.3, 'n': 212.8, 'MissV': 0}  # 碱性
            self.dicM = {'A': 71.037114, 'C': 103.009185, 'D': 115.026943, 'E': 129.042593, 'F': 147.068414,
                     'G': 57.021464, 'H': 137.058912, 'I': 113.084064, 'K': 128.094963, 'L': 113.084064,
                     'M': 131.040485, 'N': 114.042927, 'P': 97.052764, 'Q': 128.058578, 'R': 156.101111,
                     'S': 87.032028, 'T': 101.047678, 'V': 99.068414, 'W': 186.079313, 'Y': 163.063329,
                     'c': 160.0306486796, 'm': 147.035399708, 'n': 115.026943025, 'MissV': 0}   #质量
            self.dicS = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
                     'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
                     'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
                     'c': 20, 'm': 21, 'n': 22, 'MissV': -1}     #氨基酸编号
            self.dicHe = {'A': 1.24, 'C': 0.79, 'D': 0.89, 'E': 0.85, 'F': 1.26, 'G': 1.15, 'H': 0.97,
                      'I': 1.29, 'K': 0.88, 'L': 1.28, 'M': 1.22, 'N': 0.94, 'P': 0.57, 'Q': 0.96,
                      'R': 0.95, 'S': 1.00, 'T': 1.09, 'V': 1.27, 'W': 1.07, 'Y': 1.11,
                      'c': 0.79, 'm': 1.22, 'n': 0.94, 'MissV': 0}  # 螺旋性
            self.dicHy = {'A': 0.16, 'C': 2.50, 'D': -2.49, 'E': -1.50, 'F': 5.00, 'G': -3.31, 'H': -4.63, 'I': 4.41,
                      'K': -5.00, 'L': 4.76, 'M': 3.23, 'N': -3.79, 'P': -4.92, 'Q': -2.76, 'R': -2.77, 'S': -2.85,
                      'T': -1.08, 'V': 3.02, 'W': 4.88, 'Y': 2.00,
                      'c': 2.50, 'm': 3.23, 'n': -3.79, 'MissV': 0}  # 疏水性
            self.dicP = {'A': 6.02, 'C': 5.02, 'D': 2.97, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59, 'I': 6.02,
                     'K': 9.74, 'L': 5.98, 'M': 5.75, 'N': 5.41, 'P': 6.30, 'Q': 5.65, 'R': 10.76, 'S': 5.68,
                     'T': 6.53, 'V': 5.97, 'W': 5.89, 'Y': 5.66,
                     'c': 5.02, 'm': 5.75, 'n': 5.41, 'MissV': 0}  # 等电点

            self.PROTON= 1.007276466583
            self.H = 1.0078250322
            self.O = 15.9949146221
            self.H2O=self.H * 2 + self.O

        self.prev = 1
        self.next = 1
        self.aa2vector = self.AAVectorDict()
        self.AA_idx = dict(zip("ACDEFGHIKLMNPQRSTVWY",range(0,len(self.aa2vector))))

    def AAVectorDict(self):
        aa2vector_map = {}
        s = "ACDEFGHIKLMNPQRSTVWY"
        v = [0]*len(s)
        v[0] = 1
        for i in range(len(s)):
            aa2vector_map[s[i]] = list(v)
            v[i],v[(i+1) % 20] = 0,1
        return aa2vector_map

    def get_features_2_norm_105(self,line,fragmentation_window_size = 1 ):
        features_list = []
        line = line.replace('\n','').split('\t')
        peptide = line[0]
        charge = int(line[1])
        b_ion = line[2].split(',')[0]
        y_ion = line[2].split(',')[1]
        peptide_list = []
        for i in range(fragmentation_window_size-1):
            peptide_list.append('MissV')
        peptide_list.extend(list(peptide))
        for i in range(fragmentation_window_size-1):
            peptide_list.append('MissV')

        # 获取碎裂点左右各fragmentation_window_siez个氨基酸
        len_b = len(b_ion)
        r_l_fragmentation = peptide_list[len_b-1:len_b-1+2*fragmentation_window_size]

        # 碎裂窗口
        temp_features=[0]*23*2*fragmentation_window_size
        j = 0
        for i in range(len(r_l_fragmentation)):
            aa_index = self.dicS[r_l_fragmentation[i]]
            if aa_index != -1:
                temp_features[j*23 + aa_index] = 1
            j += 1
        features_list.extend(temp_features)

        # C端，N端肽的身份
        temp_features=[0]*46
        temp_features[self.dicS[peptide[0]]] = 1
        temp_features[23 + self.dicS[peptide[-1]]] = 1
        features_list.extend(temp_features)

        # 碎裂点是否在肽的一端
        if len(b_ion) == 1:
            features_list.extend([1])
        else:
            features_list.extend([0])

        # 肽和b/y离子中碱性氨基酸的数量
        features_list.extend([peptide.count('K') + peptide.count('R') + peptide.count('H')])
        features_list.extend([b_ion.count('K') + b_ion.count('R') + b_ion.count('H')])
        features_list.extend([y_ion.count('K') + y_ion.count('R') + y_ion.count('H')])

        # 碎裂点距离肽N端和C端的距离
        features_list.extend([len(b_ion)])
        features_list.extend([len(peptide)-len(b_ion)])

        # 肽序列的长度
        features_list.extend([len(peptide)])

        # 肽带电量
        p_charge = [0]*5
        p_charge[charge-1] = 1
        features_list.extend(p_charge)

        #肽的电子迁移率  Charge-Arg-0.5*(His+Lys)
        Mob = charge - peptide_list.count('R') - 0.5 * (peptide_list.count('H') + peptide_list.count('K'))
        features_list.extend([Mob])

        # 标签值
        if self.label_number == 2:
            norm_intensity = line[6].split(',')[0:3:2]
            label = [0]*2
            label = list(map(float,norm_intensity))
        elif self.label_number == 4:
            norm_intensity = line[6].split(',')
            label = [0]*4
            label = list(map(float,norm_intensity))
        return features_list,label
        ## 105

    def label_discretization(self,label,length):
        print('[data processing]start label discretization !')
        if self.discretization == 0:        ##等宽离散
            label = np.ceil(np.array(label)*10).astype(int).tolist()      ##区间宽度 0.1
            # label = np.ceil(np.array(label)*20).astype(int).tolist()        ##区间宽度 0.05
        if self.discretization == 1:        ##等频离散
            class_number = 9
            w = [1.0 * i / class_number for i in range(class_number)]
            p_label = pd.DataFrame(np.array(label).reshape(-1))
            if self.run_model == 'Train':
                _p_label = p_label[~p_label[0].isin([0])]       ##除开0，对其他数值进行等频离散
                print(_p_label.describe(w))
                w = _p_label.describe(w)[4:4+class_number+2]
                p_label = np.array(p_label).reshape(-1)
                w = np.array(w).reshape(-1)
                w[0] = w[0] - 1e-10
                print('标签离散化分段值（单独计算0）： ' + str(w))
                self.discretization_f = w
            elif self.run_model == 'Test':
                p_label = np.array(p_label).reshape(-1)
                w = self.discretization_f
            label = pd.cut(p_label,w[:],labels=[1,2,3,4,5,6,7,8,9,10],right=True)
            label = pd.DataFrame(np.array(label))
            label.fillna(0,inplace=True)
            if self.label_number == 2:
                label = np.array(label,dtype=np.int32).reshape(-1,2)
            else:
                label = np.array(label,dtype=np.int32).reshape(-1,4)
            label = label.tolist()
        if self.discretization == 2:    ## 未进行离散化
            pass
        d_label = []
        start = 0
        for l in length:
            d_label.append(label[start:start+l[0]])
            start += l[0]
        print('[data processing]label discretization end !')
        return d_label

    def get_level01(self,norm_intensity):
        i = float(norm_intensity)
        level = 0
        if i>0:
            level= 1
        return level

    def get_batch(self,data,label,length,batch_size):
        batch_data = []
        batch_label = []
        batch_length = []
        batch_index = []
        print('[data processing]start get batch !')
        data_size = len(data)
        if self.run_model == 'Train':
            p = np.random.permutation(data_size).tolist()
        else:
            p = np.linspace(0,data_size-1,data_size,dtype=int).tolist()
        batch_count = 0
        end = 0
        start = 0
        while (end + batch_size-data_size) < batch_size :
        # while (end+batch_size) < data_size:
            start = batch_size*batch_count
            batch_count += 1
            end = start + batch_size
            temp_index = p[start:end]
            temp_data = []
            temp_label = []
            temp_length = []
            batch_index.append(temp_index)
            for i in temp_index:
                temp_data.append(data[i])
                temp_label.append(label[i])
                temp_length.extend(length[i])
            batch_data.append(temp_data)
            batch_label.append(temp_label)
            batch_length.append(temp_length)
        print('[data processing]get batch end !')
        return batch_data,batch_label,batch_length,batch_index

    def padding(self,batch_data,batch_label,batch_length,batch_size):
        for index in range(len(batch_data)):
            batch_max_len = max(batch_length[index])
            for l in range(len(batch_length[index])):
                if batch_length[index][l] < batch_max_len:
                    for ii in range(batch_max_len-batch_length[index][l]):
                        batch_data[index][l].append([0.0] * len(batch_data[index][0][0]))
                        if self.label_number == 2:
                            batch_label[index][l].append([-1,-1])
                        else:
                            batch_label[index][l].append([-1,-1,-1,-1])
        return batch_data,batch_label,batch_length

    def data_Standardization(self,data,length):
        print('[data processing]start data standardization !')
        # data = np.array(data)
        if self.Standardization:
            # scaler = StandardScaler()
            # trans_data = scaler.fit_transform(data)
            scaler = MaxAbsScaler()
            data = scaler.fit_transform(data)
        else:
            # data = data
            pass
        _data = []
        start = 0
        for l in length:
            _data.append(data[start:(start+l[0])])
            start += l[0]
        print('[data processing]data standardization end !')
        return _data

    def storage_data(self,data,label,length,v_data,v_label,v_length,nce,charge):
        if os.path.exists('./_data_' + str(self.label_number)+'_nce_'+str(nce)+'_charge_'+str(charge)):
            pass
        else:
            os.mkdir('./_data_' + str(self.label_number)+'_nce_'+str(nce)+'_charge_'+str(charge))
        np.savez('./_data_' + str(self.label_number)+'_nce_'+str(nce)+'_charge_'+str(charge) + '/_data_train'+'_'+str(charge) ,data = np.array(data),label = np.array(label),length = np.array(length))
        np.savez('./_data_' + str(self.label_number)+'_nce_'+str(nce)+'_charge_'+str(charge) + '/_data_validation'+'_'+str(charge) ,data = np.array(v_data),label = np.array(v_label),length = np.array(v_length))
        print('[data processing]have storaged data !')

    def read_data(self,train_file):
        with open(self.workpath + '/' + train_file) as r:
            _content = []
            _label = []
            _length = []
            line = r.readline()
            while True:
                if not line.strip('\n'):
                    break
                pairs_count = len(line.split('\t')[0]) - 2
                i = 0
                one_seq_content = []
                one_seq_label = []
                one_seq_length = [pairs_count + 1]
                while i <= pairs_count and line.strip('\n'):
                    features, labels = self.get_features_2_norm_105(line)
                    one_seq_content.extend([features])
                    one_seq_label.extend([labels])
                    line = r.readline()
                    i += 1
                _content.extend(one_seq_content)
                _label.extend(one_seq_label)
                _length.append(one_seq_length)
            print('[data processing]read data end !')
            return _content, _label, _length

    def GetData(self,batch_size):
        print('[data processing]data processing model: ' + self.run_model)
        if self.run_model == 'Train':
            print('[data processing]Start processing data files!')
            train_content,train_label,train_length = self.read_data(self.train_file)
            validation_content,validation_label,validation_length = self.read_data(self.validation_file)
            train_label = self.label_discretization(train_label, train_length)
            train_content = self.data_Standardization(train_content, train_length)
            validation_label = self.label_discretization(validation_label,validation_length)
            validation_content = self.data_Standardization(validation_content,validation_length)
            train_content, train_label, train_length, batch_index = self.get_batch(train_content, train_label,train_length, batch_size)
            validation_content,validation_label,validation_length,v_batch_index = self.get_batch(validation_content,validation_label,validation_length,batch_size)
            train_content, train_label, train_length = self.padding(train_content, train_label, train_length, batch_size)
            validation_content, validation_label, validation_length = self.padding(validation_content,validation_label,validation_length,batch_size)
            _l = int(len(train_content)*0.8)
            train_content = train_content[:_l]
            train_label = train_label[:_l]
            train_length = train_length[:_l]
        elif self.run_model == 'Test':
            train_content,train_label,train_length = self.read_data(self.test_file)
            train_label = self.label_discretization(train_label, train_length)
            train_content = self.data_Standardization(train_content, train_length)
            train_content, train_label, train_length, batch_index = self.get_batch(train_content, train_label,train_length, batch_size)
            train_content, train_label, train_length = self.padding(train_content, train_label, train_length, batch_size)
            validation_content = []
            validation_label = []
            validation_length = []
        return train_content,train_label,train_length,validation_content,validation_label,validation_length
    # format:btach_size,seq_length,features_number

if __name__ == '__main__':
    test = data('E:/data/1/test',2,discretization=0)
    Train_data,Train_label,Train_length,Test_data,Test_label,Test_length = test.GetData(3)
    print(len(Train_data[0][0]))
    print(Train_label[0])
    print(Train_length)

    # print(Test_index)
    # print(len(Train_data[0][0][1]))
    # sentence = [Train_data[0][i][1::2] for i in range(3)]
    # label = [Train_data[0][i][2::2] for i in range(3)]
    # print(sentence)
    # print(label)
    # print(len(Train_data[0]))
    # s = test.get_features_2_norm('AKHAVSEGTKAVTKYMc	2	A,KHAVSEGTKAVTKYMc	*	b1+,b1++,y14+,y14++	0.0,0.0,0.0,0.0	0.0,0.0,0.0,0.0	0.0,0.0,0.0,0.0')
    # print(s[0][:141] + StandardScaler().fit_transform(np.array(s[0][141:]).reshape(-1,1)).reshape(-1).tolist())
    # print(s)
    # print(sorted(test.dicP.items(),key=lambda item:item[1]))
