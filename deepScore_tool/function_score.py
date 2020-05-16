# -*- coding: utf-8 -*-
# @Software: PyCharm
# @File: function_score.py
# @Author: MX
# @E-mail: minxinm@foxmail.com
# @Time: 2020/2/21
import os

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from Model.Resnet_model import ResNet18
from deepScore_tool.data_util_tool import data
from deepScore_tool.match_ions_tool import MATCH


class ScoreEngine(object):
    def __init__(self, peptidefile, spectrumfile, NCE, inputformat, ppm, t_fdr, outputformat, addstates, Status):
        """
        实例初始化，完成质谱文件读取以及候选肽读取
        :param peptidefile:
        :param spectrumfile:
        :param NCE:
        :param inputformat:
        :param ppm:
        :param t_fdr:
        :param outputformat:
        :param addstates:
        :param Status:
        """
        self.spectrumfile = spectrumfile
        self.NCE = NCE
        self.input_format = inputformat
        self.ppm_threshold = ppm
        self.fdr_threshold = t_fdr
        self.output_fileformat = outputformat
        self.add_states = addstates
        self.Status = Status
        if self.input_format == 'MSGF+':
            self.peptides = self.readpeptide_msgf(peptidefile, have_decoy=True)
        elif self.input_format == 'Comet':
            self.peptides = self.readpeptide_comet(peptidefile, have_decoy=True)
        else:
            print('aaa' + str(self.input_format))
            self.peptides = self.readpeptide_customize(peptidefile)
        self.spectrums = self.readspectrum(spectrumfile)
        for f in os.listdir('./data'):
            if f in ['allPSMs.mgf', 'allPSMs_byions.txt', 'FDR.txt', 'FDR_plot.png', 'PSMs_score.txt']:
                os.remove('./data/' + f)

    def readpeptide_customize(self, filepath):
        """
        读取自定义候选肽序列存储文件并以字典形式返回
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
                index, peptide = line
                if peptides.get(index):
                    peptides[index].append(peptide)
                else:
                    peptides[index] = [peptide]
        return peptides

    def readpeptide_msgf(self, filepath, have_decoy=False, have_score=False, have_charge=False):
        """
        读取来自MSGF+的鉴定结果并将候选肽以字典形式返回
        :param filepath:
        :param have_decoy:
        :param have_score:
        :param have_charge:
        :return:
        """
        with open(filepath, 'r', encoding='utf-8') as rf:
            _results = {}
            CHARGE = {}
            _flag = 0
            _ = rf.readline()
            while True:
                line = rf.readline()
                if line.strip() == '':
                    break
                l = line.strip().split('\t')
                _index = str(int(l[1].split('=')[1]) + 1)
                _charge = l[8]
                if int(_charge) > 6:
                    _charge = '6'
                _evalue = l[14]
                _seq = l[9][2:-2]
                _M = []
                _seq = _seq.replace('+15.995', 'm')
                _seq = _seq.replace('+57.021', 'c')
                if '+' in _seq or 'U' in _seq or 'X' in _seq:
                    continue
                if 'c' in _seq:
                    _C = ';Carbamidomethyl@C'
                    _seq = _seq.replace('c', '')
                else:
                    _C = ';'
                while 'm' in _seq:
                    _m_index = _seq.index('m')
                    _M.append('Oxidation@M' + str(_m_index))
                    _seq = _seq.replace('m', '', 1)
                if 'Decoy_' in l[10]:
                    _seq = 'DECOY-' + _seq
                    if not have_decoy:
                        continue
                _modif = ';'.join(_M) + _C
                if _results.get(_index) == None:
                    if have_score:
                        _results[_index] = [[_seq + '_' + _modif, _evalue]]
                    else:
                        _results[_index] = [_seq + '_' + _modif]
                    if have_charge:
                        CHARGE[_index] = _charge
                elif _results.get(_index) != None:
                    if have_score:
                        for i in _results[_index]:
                            _s = _seq + '_' + _modif
                            if i[0] == _s:
                                _flag = 1
                        if _flag == 1:
                            _flag = 0
                            continue
                        _results[_index].append([_seq + '_' + _modif, _evalue])
                    else:
                        for i in _results[_index]:
                            _s = _seq + '_' + _modif
                            if i == _s:
                                _flag = 1
                        if _flag == 1:
                            _flag = 0
                            continue
                        _results[_index].append(_seq + '_' + _modif)
            print('MSGF+ results number : ' + str(len(_results)))
            if have_charge:
                return _results, CHARGE
            else:
                return _results

    def readpeptide_comet(self, filepath, have_decoy=False, have_score=False, have_charge=False):
        """
        读取来自Comet的鉴定结果并将候选肽以字典形式返回
        :param filepath:
        :param have_decoy:
        :param have_score:
        :param have_charge:
        :return:
        """
        with open(filepath, 'r', encoding='utf-8') as rf:
            rf.readline()
            rf.__next__()
            results = {}
            CHARGE = {}
            while True:
                line = rf.readline().strip().split('\t')
                if line == ['']:
                    break
                Index = line[0]
                sequence = line[11]
                if 'U' in sequence or 'X' in sequence:
                    continue
                _charge = line[2]
                modif = line[17]
                if 'DECOY_' in line[15].split(',')[0]:
                    if have_decoy:
                        sequence = 'DECOY-' + line[11]
                    else:
                        continue
                if modif != '-':
                    modif = modif.split(',')
                    _M = []
                    _C = ''
                    for one in modif:
                        _one_modif = one.split('_')
                        if _one_modif[1] == 'V':
                            _M.append('Oxidation@M' + _one_modif[0])
                        else:
                            if ';Carbamidomethyl@C' == _C:
                                continue
                            else:
                                _C = ';Carbamidomethyl@C'
                    if _C == '':
                        _C = ';'
                    _M = sorted(_M, key=lambda x: int(x.split('@M')[1]))
                    _modif = ';'.join(_M) + _C
                    modif = _modif
                else:
                    modif = ';'
                if results.get(Index) == None:
                    if have_score:
                        results[Index] = [[sequence + '_' + modif, line[5]]]
                    else:
                        results[Index] = [sequence + '_' + modif]
                    if have_charge:
                        CHARGE[Index] = _charge
                elif results.get(Index) != None:
                    if have_score:
                        results[Index].append([sequence + '_' + modif, line[5]])
                    else:
                        results[Index].append(sequence + '_' + modif)
        print('Comet results number : ' + str(len(results)))
        if have_charge:
            return results, CHARGE
        else:
            return results

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
        assert len(self.peptides) == len(self.spectrums), '候选肽数量与质谱谱图数量不一致！'
        self.Status.SetStatusText('正在根据候选肽和质谱生成PSM ...')
        with open('./data/allPSMs.mgf', 'w', encoding='utf-8') as fw:
            for i in range(len(self.spectrums)):
                temp_peptides = self.peptides[str(i + 1)]
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
        self.Status.SetStatusText('碎片离子标注中 ...')
        m = MATCH('./data', 'allPSMs.mgf', self.ppm_threshold)
        m.write_files()

    def caculateScore(self):
        """
        加载模型输出概率矩阵,利用概率矩阵和真实标签计算出deepScore-α分数并生成分数文件
        :param NCE:
        :return:
        """
        self.Status.SetStatusText('正在进行分数计算 ...')
        ##Model parameters
        BATCH_SIZE = 16
        Label_number = 4
        features_size = 105
        weights4 = [0.6550, 0.1455, 0.0718, 0.0385, 0.0236, 0.0158, 0.0113, 0.0084, 0.0063, 0.0047, 0.0190]
        #   Run
        print('start caculate score ...')
        model = ResNet18(BATCH_SIZE, weight=weights4, feature_size=features_size)
        model.load_state_dict(torch.load('./model_trained/NCE%s_model.pkl' % (self.NCE)))
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
            allPsms_score = []
            keys = sorted(list(map(int, self.peptides.keys())))
            candidats = []
            for k in keys:
                for p in self.peptides[str(k)]:
                    candidats.append([str(k), p])
            while start + 2 <= len(Results):
                _p = P[int(start / 2)]
                _matrix_p = Matrix_P[int(start / 2)]
                _true = Results[start]
                _pred = Results[start + 1]
                _score = 1.0
                for i in range(len(_matrix_p)):
                    _p = _matrix_p[i]
                    _score = _score + _p[_true[i]]
                PQC = ((len(_true) - _true.count(0) + 1) / (len(_true) + 1))
                _score = _score * PQC
                allPsms_score.append([candidats[int(start / 2)][0], candidats[int(start / 2)][1], str(_score)])
                start += 2
            if self.add_states['isScorefile']:
                print('start to write PSMs score ...')
                with open('./data/PSMs_score.txt', 'a+', encoding='utf-8') as fw:
                    for l in allPsms_score:
                        line = '%s\t%s\t%s\n' % (l[0], l[1], l[2])
                        fw.write(line)
                print('write results end!')
        self.Status.SetStatusText('分数计算完毕')
        return allPsms_score

    def write_identity(self, allPsms_score, Scorethreshold):
        """
        根据分数计算结果以及获得的分数阈值写入符合条件的鉴定结果
        :param allPsms_score:
        :param Scorethreshold:
        :return:
        """
        self.Status.SetStatusText('开始写入鉴定结果...')
        with open('./data/identity_results.%s' % (self.output_fileformat), 'w', encoding='utf-8') as fw:
            identity_results = {}
            for l in allPsms_score:
                if identity_results.get(l[0]) == None:
                    identity_results[l[0]] = [l]
                else:
                    identity_results[l[0]].append(l)
            keys = sorted(list(identity_results.keys()), key=lambda x: int(x))
            for k in keys:
                temp_results = sorted(identity_results[k], key=lambda x: float(x[2]), reverse=True)
                top1 = temp_results[0]
                if float(top1[2]) >= float(Scorethreshold):
                    fw.write('%s\t%s\t%s\n' % (top1[0], top1[1], top1[2]))
        print('鉴定结果写入完毕')
        self.Status.SetStatusText('鉴定结果写入完毕')

    def caculateFDR(self, allPsms_score):
        """
        通过PSMs打分计算出FDR值，再根据给定FDR阈值计算出分数阈值，写入鉴定结果，并按需求绘制FDR曲线图
        :param allPsms_score:
        :return:
        """
        self.Status.SetStatusText('正在计算FDR值 ...')
        PSMs_score = []
        for p in allPsms_score:
            PSMs_score.append(float(p[2]))
        Charges = []
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
        indexs = sorted(list(self.peptides.keys()), key=lambda x: int(x))
        start = 0
        pep_score = []
        threshold_score = []
        for i in indexs:
            temp = []
            for pep in self.peptides[i]:
                temp.append([pep, PSMs_score[start]])
                start += 1
            temp = sorted(temp, key=lambda x: x[1], reverse=True)
            top_pepscore = temp[0]
            top_pepscore.append(Charges[int(i) - 1])
            pep_score.append(top_pepscore)
            if temp[0][1] not in threshold_score:
                threshold_score.append(round(temp[0][1], 4))
        threshold_score = sorted(threshold_score)
        score_fdr = {}
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
                    _line = 'Threshold peptide score : ' + str(t) + '\tFDR : ' + str(
                        False_Discover_Rate) + '\ttarget hits : ' + str(
                        len(threshold_seq_score)) + '\tcharge : ' + str(c)
                    fw.write(_line + '\n')

                False_seq_score = list(x for x in threshold_all if x[0].startswith('DECOY-'))
                try:
                    False_Discover_Rate = len(False_seq_score) / (len(threshold_all) - len(False_seq_score))
                except:
                    False_Discover_Rate = 0.0

                if score_fdr.get(False_Discover_Rate) == None:
                    score_fdr[False_Discover_Rate] = t
                else:
                    if t > score_fdr[False_Discover_Rate]:
                        score_fdr[False_Discover_Rate] = t

        fdr_dif = abs(np.array(list(score_fdr.keys())) - 1)
        fdr_index = np.argmin(fdr_dif)
        Score_threshold = score_fdr[list(score_fdr.keys())[fdr_index]]

        self.Status.SetStatusText('FDR计算完毕')
        if self.add_states['isFdrplot']:
            self.plot_qvalue()

        self.write_identity(allPsms_score, Score_threshold)

    def get_qvalue(self, file='./data/FDR.txt', charge_number=3):
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
        q_values, target_hits = self.get_qvalue()
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
        # plt.show()
        self.Status.SetStatusText('FDR曲线图绘制完成，已保存到data目录下')


if __name__ == '__main__':
    peptidefile = './data/test_peptide.txt'
    spectrumfile = './data/test_spectrum.mgf'
    NCE = '30'
    test = ScoreEngine(peptidefile, spectrumfile, NCE)

    test.productPSMs()
    test.caculateScore()
    test.caculateFDR_Plot()
