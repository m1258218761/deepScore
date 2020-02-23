# -*- coding: utf-8 -*-
# @Software: PyCharm
# @File: function_score.py
# @Author: MX
# @E-mail: minxinm@foxmail.com
# @Time: 2020/2/21
import os

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_score import data
from match_ions import MATCH
from Resnet_model import ResNet18

class ScoreEngine(object):
    def __init__(self, peptidefile, spectrumfile, NCE):
        """
        实例初始化，读取候选肽以及质谱谱图
        :param peptidefile:
        :param spectrumfile:
        """
        self.spectrumfile = spectrumfile
        self.NCE = NCE
        self.peptides = self.readpeptide(peptidefile)
        self.spectrums = self.readspectrum(spectrumfile)
        for f in os.listdir('./data'):
            if f in ['allPSMs.mgf','allPSMs_byions.txt','FDR.txt','FDR_plot.png','PSMs_score.txt']:
                os.remove('./data/'+f)

    def readpeptide(self, filepath):
        """
        读取候选肽序列存储文件并以字典形式返回
        :param filepath:
        :return:{'1':[peptide1,peptide2,...],'2':[peptide1,peptide2,...],...}
        """
        peptides = {}
        with open(filepath, 'r', encoding='utf-8') as fr:
            while True:
                line = fr.readline().strip()
                if not line:
                    break
                line = line.split('\t')
                index,peptide = line
                if peptides.get(index):
                    peptides[index].append(peptide)
                else:
                    peptides[index] = [peptide]
        return peptides

    def readspectrum(self, filepath):
        """
        读取质谱谱图存储文件并以列表形式返回
        :param filepath:
        :return:[[spectrum1],[spectrum2],....]
        """
        with open(filepath, 'r', encoding='utf-8') as fr:
            spectrums = []
            while True:
                line = fr.readline()
                if not line:
                    break
                _line = []
                if 'BEGIN IONS' in line:
                    _line.append(line)
                    while True:
                        line = fr.readline()
                        _line.append(line)
                        if 'END IONS' in line:
                            spectrums.append(_line)
                            break
        print('spectrum number : ' + str(len(spectrums)))
        return spectrums

    def productPSMs(self):
        """
        形成PSMs文件并对所有PSM进行离子标注，生成标注结果文件，用于模型输入
        :param peptidefile:
        :param spectrumfile:
        :return:
        """
        assert len(self.peptides) == len(self.spectrums),'候选肽数量与质谱谱图数量不一致！'
        with open('./data/allPSMs.mgf', 'w', encoding='utf-8') as fw:
            for i in range(len(self.spectrums)):
                temp_peptides = self.peptides[str(i+1)]
                for p in temp_peptides:
                    _line = []
                    if p.startswith('DECOY-'):
                        _pep = p.split('-')[1].split('_')[0]
                    else:
                        _pep = p.split('_')[0]
                    _Modified = p.split('_')[1]
                    _spectrum = self.spectrums[i]
                    for _l in _spectrum:
                        if _l.startswith('CHARGE='):
                            _l = _l.replace('+', '')
                            _line.append('Sequence=' + _pep + '\n')
                        if _l.startswith('RTINSECONDS='):
                            _line.append('Modified=' + _Modified + '\n')
                        if _l.startswith('PIF='):
                            continue
                        _line.append(_l)
                    fw.write(''.join(_line))
        m = MATCH('./data','allPSMs.mgf')
        m.write_files()

    def caculateScore(self):
        """
        加载模型输出概率矩阵,利用概率矩阵和真实标签计算出deepScore-α分数并生成分数文件
        :param NCE:
        :return:
        """
        ##Model parameters
        BATCH_SIZE = 16
        Label_number = 4
        features_size = 105
        weights4 = [0.6550, 0.1455, 0.0718, 0.0385, 0.0236, 0.0158, 0.0113, 0.0084, 0.0063, 0.0047, 0.0190]
        #   Run
        print('start...')
        model = ResNet18(BATCH_SIZE, weight=weights4, feature_size=features_size)
        model.load_state_dict(torch.load('./model_trained/NCE%s_model.pkl'%(self.NCE)))
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        test_data = data('./data', Label_number, run_model='Test', test_file='allPSMs_byions.txt')
        Test_data, Test_label, Test_length, _, _, _ = test_data.GetData(BATCH_SIZE)
        print('data number: ' + str(len(Test_length) * BATCH_SIZE))
        with torch.no_grad():
            Results = []
            P = []
            Matrix_P = []
            for T_index, T_data in tqdm(enumerate(Test_data)):
                t_data = T_data
                t_label = Test_label[T_index]
                t_length = Test_length[T_index]
                t_input_features = torch.tensor(t_data).cuda()
                t_ions_level = torch.tensor(t_label).cuda()
                t_batch_length = torch.tensor(t_length).cuda()
                y_true, y_pred, results, loss, _p = model(t_input_features.permute(0, 2, 1), t_ions_level,
                                                          t_batch_length)
                Results.extend(results)
                P.extend(_p[0])
                Matrix_P.extend(_p[1])

            start = 0
            print('start to write results...')
            with open('./data/PSMs_score.txt', 'a+') as fw:
                keys = sorted(list(map(int,self.peptides.keys())))
                candidats = []
                for k in keys:
                    for p in self.peptides[str(k)]:
                        candidats.append([str(k),p])
                while start + 2 <= len(Results):
                    _p = P[int(start / 2)]
                    _matrix_p = Matrix_P[int(start / 2)]
                    _true = Results[start]
                    _pred = Results[start + 1]
                    _score = 1.0
                    for i in range(len(_matrix_p)):
                        _p =  _matrix_p[i]
                        _score = _score + _p[_true[i]]
                    PQC = ((len(_true) - _true.count(0) + 1) / (len(_true) + 1))
                    _score = _score * PQC
                    line = '%s\t%s\t%s\n'%(candidats[int(start/2)][0], candidats[int(start/2)][1], str(_score))
                    fw.write(line)
                    start += 2
            print('write results end!')

    def caculateFDR_Plot(self):
        """
        通过PSMs打分计算出FDR值以及对应的target hits数量,并绘制FDR曲线图
        :param peptidefile:
        :return:
        """
        PSMs_score = []
        with open('./data/PSMs_score.txt', 'r') as fr:
            while True:
                line = fr.readline().strip()
                if not line:
                    break
                PSMs_score.append(float(line.split('\t')[2]))
        Charges= []
        with open(self.spectrumfile, 'r') as fr:
            while True:
                line = fr.readline().strip()
                if not line:
                    break
                if line.startswith('CHARGE='):
                    charge = line.split('=')[1][0]
                    if int(charge) > 4:
                        charge = 4
                    Charges.append(charge)
        indexs = sorted(list(self.peptides.keys()), key=lambda x:int(x))
        start = 0
        pep_score = []
        threshold_score = []
        for i in indexs:
            temp = []
            for pep in self.peptides[i]:
                temp.append([pep,PSMs_score[start]])
                start += 1
            temp = sorted(temp, key=lambda x:x[1], reverse=True)
            top_pepscore = temp[0]
            top_pepscore.append(Charges[int(i)-1])
            pep_score.append(top_pepscore)
            if temp[0][1] not in threshold_score:
                threshold_score.append(round(temp[0][1],4))
        threshold_score = sorted(threshold_score)
        with open('./data/FDR.txt', 'a+', encoding='utf-8') as fw:
            for _index in tqdm(range(len(threshold_score))):
                t = threshold_score[_index]
                threshold_all = list(x for x in pep_score if x[1] >= t)
                for c in ['2', '3', '4']:
                    threshold_seq_score = list(x for x in threshold_all if x[2] == c)
                    False_seq_score = list(x for x in threshold_seq_score if x[0].startswith('DECOY-'))
                    try:
                        False_Discover_Rate = len(False_seq_score) / (len(threshold_seq_score) - len(False_seq_score))
                    except:
                        False_Discover_Rate = 0.0
                    _line = 'Threshold peptide score : ' + str(t) + '\tFDR : ' + str(False_Discover_Rate) + '\ttarget hits : ' + str(
                        len(threshold_seq_score)) + '\tcharge : ' + str(c)
                    fw.write(_line + '\n')

        self.plot_qvalue()

    def get_qvalue(self, file = './data/FDR.txt', charge_number = 3):
        """
        通过FDR计算q-value，返回q-value值以及对应的target hits数量
        :param file:
        :param charge_number:
        :return:
        """
        with open(file, 'r') as fr:
            _score_threshold = [[] for x in range(charge_number)]
            _FDR = [[] for x in range(charge_number)]
            _targethits = [[] for x in range(charge_number)]
            q_value = [[] for x in range(charge_number)]
            line = fr.readline()
            flag = 0
            while True:
                if not line.strip():
                    break
                line = line.strip().split('\t')
                _score_threshold[flag].append(float(line[0].split(':')[1][1:]))
                _FDR[flag].append(float(line[1].split(':')[1][1:]))
                _targethits[flag].append(int(line[2].split(':')[1][1:]))
                flag += 1
                if flag == charge_number:
                    flag = 0
                line = fr.readline()
            print('\nstart caculate q value... file : ' + file)
            for _i in tqdm(range(len(_score_threshold[0]))):
                for c in range(charge_number):
                    _temp_fdr = sorted(_FDR[c][:_i + 1])
                    _q_value = _temp_fdr[0]
                    q_value[c].append(_q_value)

            for i in range(len(q_value)):
                q_t = list(zip(q_value[i], _targethits[i]))
                q_t = [x for x in q_t if x[0] <= 0.05]
                q_value[i] = [x[0] for x in q_t]
                _targethits[i] = [x[1] for x in q_t]
        return q_value, _targethits

    def plot_qvalue(self):
        """
        绘制FDR曲线图并保存图片
        :return:
        """
        q_values,target_hits = self.get_qvalue()
        plt.figure(figsize=(15, 5), dpi=100)
        label1 = 'deepScore-α'

        fig1 = plt.subplot(1, 3, 1)
        fig1.grid()
        fig1.plot(q_values[0], target_hits[0], '-r', label=label1, linewidth=2.0)
        fig1.set_ylabel('target hits', fontsize=20)
        fig1.set_xlabel('FDR(q-value)', fontsize=20)
        fig1.legend(loc='best', fontsize='x-large')
        fig1.set_title('CHARGE 2', fontsize=20)
        fig1.axvline(0.01, 0, 1, color="#949494", linestyle="dashed")
        fig1.axvline(0.05, 0, 1, color="#949494", linestyle="dashed")
        # plt.ylim(1800)

        fig2 = plt.subplot(1, 3, 2)
        fig2.grid()
        fig2.plot(q_values[1], target_hits[1], '-r', label=label1, linewidth=2.0)
        fig2.set_ylabel('target hits', fontsize=20)
        fig2.set_xlabel('FDR(q-value)', fontsize=20)
        fig2.legend(loc='best', fontsize='x-large')
        fig2.set_title('CHARGE 3', fontsize=20)
        fig2.axvline(0.01, 0, 1, color="#949494", linestyle="dashed")
        fig2.axvline(0.05, 0, 1, color="#949494", linestyle="dashed")
        # plt.ylim(1700)

        fig3 = plt.subplot(1, 3, 3)
        fig3.grid()
        fig3.plot(q_values[2], target_hits[2], '-r', label=label1, linewidth=2.0)
        fig3.set_ylabel('target hits', fontsize=20)
        fig3.set_xlabel('FDR(q-value)', fontsize=20)
        fig3.legend(loc='best', fontsize='x-large')
        fig3.set_title('CHARGE 4', fontsize=20)
        fig3.axvline(0.01, 0, 1, color="#949494", linestyle="dashed")
        fig3.axvline(0.05, 0, 1, color="#949494", linestyle="dashed")
        # plt.ylim(500)

        plt.tight_layout()
        # plt.savefig('D:/1.svg', dpi=800, format='svg')
        plt.savefig('./data//FDR_plot.png', dpi=600)
        plt.show()



if __name__ == '__main__':
    peptidefile = './data/test_peptide.txt'
    spectrumfile = './data/test_spectrum.mgf'
    NCE = '30'
    test = ScoreEngine(peptidefile, spectrumfile, NCE)

    test.productPSMs()
    test.caculateScore()
    test.caculateFDR_Plot()
