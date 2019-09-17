#coding=utf-8
import copy
from match_ions import MATCH
import torch
from tqdm import tqdm
import numpy as np
import math
from data_util import data
from Resnet_model import ResNet18
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity

'''
This is for ProteomeTools2 dataset
'''
class ProteomeTools(object):
    def __init__(self, workpath='',nce=''):
        self.workpath = workpath+'/NCE'+nce+'/'
        self.nce = nce

    #   Compare string a and string b difference
    def find_diff(self, a, b):
        diff_index = np.array(list(a)) != np.array(list(b))
        array_a = np.array(list(a))
        diffa = list(array_a[diff_index])
        result_stra = ""
        for x in diffa:
            result_stra += x
        array_b = np.array(list(b))
        diffb = list(array_b[diff_index])
        result_strb = ""
        for x in diffb:
            result_strb += x
        return result_stra, result_strb

    #   Delete the spectrum of the specified index
    def delmore(self, index=[]):
        _index = [774, 1599, 1600, 4176]
        count = 0
        flag = 1
        with open(self.workpath+'selected_NCE'+self.nce+'.mgf', 'r') as r, open(self.workpath+'_selected_NCE'+self.nce+'.mgf', 'a+') as w:
            while True:
                line = r.readline()
                if not line.strip():
                    break
                if 'BEGIN IONS' in line:
                    count += 1
                if count in _index:
                    flag = 0
                else:
                    flag = 1
                if flag == 1:
                    w.write(line)

    # Delete the unconventional amino acids from Comet identification results:U
    def find_unkonwn_aa(self):
        with open(self.workpath+'selected_NCE'+self.nce+'_forcomet.txt','r') as r, open(self.workpath+'_selected_NCE'+self.nce+'_forcomet.txt','a+') as w:
            r.__next__()
            r.__next__()
            while True:
                line = r.readline()
                if not line.strip():
                    break
                l = line.split('\t')
                if 'U' in l[11]:
                    print(line)
                else:
                    w.write(line)

    ###---Basic function---:read Comet identification results and return
    ###   Parameter: have_decoyt:Return results include Decoy;
    ###              have_score:0 means return xcorr, 1 means return evalue;
    ###              have_charge:retrun peptide charge
    ###              filename:Comet identification file
    def read_comet_results(self, have_decoy=False, have_score=0, have_charge=False, filename=''):
        with open(filename, 'r') as rf:
            results = {}
            CHARGE = {}
            while True:
                line = rf.readline().strip().split('\t')
                if line == ['']:
                    break
                Index = line[0]
                sequence = line[11]
                _charge = line[2]
                if have_score == 0:
                    _score = line[6]  ##xcorr:6,evalue:5
                elif have_score == 1:
                    _score = line[5]
                modif = line[17]
                if 'DECOY_' in line[15].split(',')[0]:
                    if have_decoy:
                        sequence = 'DECOY-' + line[11]
                    else:
                        continue
                if modif != '-':
                    modif = modif.split(',')
                    _modif = ''
                    _M = []
                    _C = ''
                    for one in modif:
                        _one_modif = one.split('_')
                        if _one_modif[1] == 'V':
                            _M.append('Oxidation@M' + _one_modif[0])
                            # _modif.append('Oxidation@M' + _one_modif[0]+';')
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
                        results[Index] = [[sequence + '_' + modif, _score]]
                    else:
                        results[Index] = [sequence + '_' + modif]
                    if have_charge:
                        CHARGE[Index] = _charge
                elif results.get(Index) != None:
                    if have_score:
                        results[Index].append([sequence + '_' + modif, _score])
                    else:
                        results[Index].append(sequence + '_' + modif)
        print('comet results number : ' + str(len(results)))
        # print('decoy at first : ' + str(d_count))
        if have_charge:
            return results, CHARGE
        else:
            return results

    #---Basic function---:get correct peptide and spectrum
    def read_correct_PSMs(self, filename=''):
        with open(filename, 'r') as rf:
            mgf_listcontent = []
            content = []
            while True:
                line = rf.readline()
                if not line:
                    break
                _line = []
                if 'BEGIN IONS' in line:
                    _line.append(line)
                    while True:
                        line = rf.readline()
                        if 'SQE=' in line:
                            _seq = line.strip().split('=')[1]
                            _temp = _seq
                        if 'Modifications=' in line:
                            _modeified = line.strip().split('=')[1]
                            if _modeified == 'NULL':
                                _modeified = ';'
                            else:
                                _a = _modeified.split(',')[0::2]
                                _b = _modeified.split(',')[1::2]
                                _modeified = ''
                                for i in range(len(_a)):
                                    _modeified = _modeified + 'Oxidation@M' + _a[i] + ';'
                            _temp += '_' + _modeified
                            mgf_listcontent.append(_temp)
                        if 'SQE=' in line or 'Modifications=' in line or 'NCE=' in line or 'PIF=' in line or 'Score=' in line:
                            continue
                        else:
                            _line.append(line)
                        if 'END IONS' in line:
                            content.append(_line)
                            _line = []
                            break
        print('correct results number : ' + str(len(mgf_listcontent)))
        return mgf_listcontent, content

    '''-------------------------------top1 hit rate--------------------------------'''
    #   Evaluation of comet identification results and generate related files,include Comet top1 missed and unmissed
    def get_different_peptide(self):
        total_PSMs = 0
        count = 0
        unmissed_total_PSMs = 0
        unmissed_count = 0
        with open(self.workpath+'selected_NCE'+self.nce+'_missed_peptide.txt','a+') as mtw, open(self.workpath+'selected_NCE'+self.nce+'_missed_PSMs.mgf', 'a+') as mgw, open(
                self.workpath+'selected_NCE'+self.nce+'_unmissed_PSMs.mgf','a+') as ugw, open(self.workpath+'selected_NCE'+self.nce+'_unmissed_peptide.txt','a+') as utw:
            comet_results = self.read_comet_results(have_decoy=False,filename=self.workpath+'selected_NCE'+self.nce+'_forcomet.txt')
            correcte_results, correcte_spectrum = self.read_correct_PSMs(filename=self.workpath+'selected_NCE'+self.nce+'.mgf')
            for i in range(len(correcte_results)):
                correcte_seq = correcte_results[i]
                if comet_results.get(str(i + 1)) == None:
                    print('comet have no peptide index : ' + str(i + 1))
                    continue
                comet_seq = comet_results[str(i + 1)]
                c_index = 1000
                for index in range(len(comet_seq)):
                    if comet_seq[index].replace(' ', '') == correcte_seq:
                        c_index = index
                        break
                if c_index != 0:
                    mtw.write(str(i) + '\t' + correcte_seq + '\t' + '\t'.join(comet_seq) + '\n')
                    count += 1
                    comet_seq.append(correcte_seq)
                    total_PSMs += len(comet_seq)
                    for o in comet_seq:
                        seq = o.split('_')[0]
                        modif = o.split('_')[1]
                        _psm = copy.deepcopy(correcte_spectrum[i])
                        _psm.insert(2, 'Sequence=' + seq + '\n')
                        _psm.insert(4, 'Modified=' + modif + '\n')
                        mgw.write(''.join(_psm))
                if c_index == 0:
                    utw.write(str(i) + '\t' + correcte_seq + '\t' + '\t'.join(comet_seq) + '\n')
                    unmissed_count += 1
                    unmissed_total_PSMs += len(comet_seq)
                    for o in comet_seq:
                        seq = o.split('_')[0]
                        modif = o.split('_')[1]
                        _psm = copy.deepcopy(correcte_spectrum[i])
                        _psm.insert(2, 'Sequence=' + seq + '\n')
                        _psm.insert(4, 'Modified=' + modif + '\n')
                        ugw.write(''.join(_psm))
            print('missed peptide number : ' + str(count))
            print('missed total PSMs : ' + str(total_PSMs))
            print('unmissed peptide number : ' + str(unmissed_count))
            print('unmissed total PSMs : ' + str(unmissed_total_PSMs))

    #   Annotate regular ions(b1+,y1+,b2+,y2+) and generate the files can be scored by P-score
    def get_byions(self):
        m = MATCH(self.workpath, 'selected_NCE'+self.nce+'_missed_PSMs.mgf')
        m.write_files()
        um = MATCH(self.workpath, 'selected_NCE'+self.nce+'_unmissed_PSMs.mgf')
        um.write_files()

    #   Obtaining Probability Matrix by Model
    def get_MatrixP(self):
        file_mode = 'missed'
        file = 'selected_' +self.nce + '_'+file_mode+'_PSMs_byions.txt'
        ##Model parameters
        BATCH_SIZE = 16
        Label_number = 4
        features_size = 105
        weights4_nce30 = [0.5381, 0.2366, 0.0912, 0.0448, 0.0261, 0.0162, 0.0109, 0.0078, 0.0055, 0.004, 0.0187]
        weights4_nce35 = [0.6586, 0.1741, 0.0663, 0.0324, 0.0188, 0.012, 0.0083, 0.006, 0.0044, 0.0033, 0.0158]
        #   Run Testing
        print('start...')
        model = ResNet18(BATCH_SIZE, weight=weights4_nce30, feature_size=features_size)
        model.load_state_dict(torch.load('./Model/model_2_bestacc.pkl'))
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        Data = data(self.workpath+'FDR/splited_by_ions', Label_number,run_model='Test',test_file=file)
        Test_data, Test_label, Test_length, _, _, _ = Data.GetData(BATCH_SIZE)
        print('Test data number: ' + str(len(Test_length) * BATCH_SIZE))
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
                y_true, y_pred, results, loss, _p = model(t_input_features.permute(0, 2, 1), t_ions_level,t_batch_length)
                Results.extend(results)
                P.extend(_p[0])
                Matrix_P.extend(_p[1])
            start = 0
            R = []
            Mae = []
            Mae_local = []
            Cosine = []
            Cosine_0_rate = []
            print('[Score Info]start to write results...')
            # with open( self.workpath+'/35_random10000/pep_credibility/sorted_by_pccandother/' + file_mode + '_score_pep_P.txt',
            #                         'a+') as fw:
            with open(
                    self.workpath+'FDR/selected_'+self.nce+'_score_pep_P_4label_humanmodel.txt',
                    'a+') as fw:

                while start + 2 <= len(Results):
                    _p = P[int(start / 2)]
                    _matrix_p = Matrix_P[int(start / 2)]
                    _R = r2_score(Results[start], Results[start + 1])
                    R.append(_R)
                    _mae = sum(abs(np.array(Results[start]) - np.array(Results[start + 1]))) / len(Results[start])
                    Mae.append(_mae)
                    local_index = np.where((np.array(Results[start]) + np.array(Results[start + 1])) != 0)
                    try:
                        _mae_local = sum(abs(
                            np.array(Results[start])[local_index] - np.array(Results[start + 1])[local_index])) / len(
                            local_index[0])
                    except:
                        _mae_local = 0.0
                    Mae_local.append(_mae_local)
                    _Cosine = cosine_similarity([Results[start], Results[start + 1]])[0, 1]
                    Cosine.append(_Cosine)
                    _Cosine_0_rate = _Cosine * (1 - (Results[start].count(0) / len(Results[start])))
                    Cosine_0_rate.append(_Cosine_0_rate)
                    _true = ','.join(map(str, Results[start]))
                    _pred = ','.join(map(str, Results[start + 1]))
                    fw.write(
                        _true + '\t' + _pred + '\t' + str(_Cosine) + '\t' + str(_R) + '\t' + str(_mae) + '\t' + str(
                            _mae_local) + '\t' + str(_Cosine_0_rate) + '\t' + str(_p) + '\t' + str(_matrix_p) + '\n')
                    start += 2

    #   Compare Comet and P-score top1 hits rate
    def eval_prediction(self):
        file_mode = 'unmissed'
        all_pepscore = []
        all_correct_pep = []
        pre_index = []
        org_index = []
        with open(self.workpath+'selected_NCE'+self.nce+'_' + file_mode + '_peptide.txt','r') as mr, open(self.workpath+'sorted_by_pccandother/' + file_mode + '_score_pep_P.txt','r') as fr:
            score = []
            print('start reading score...')
            while True:
                line = fr.readline()  ##True,Pred,Cosine,R,mae,mae_local
                if not line.strip():
                    break
                line = line.strip().split('\t')
                y_true = list(map(int, line[0].split(',')))
                y_pred = list(map(int, line[1].split(',')))
                _score = 1.0
                matrix_p = line[8].strip()[2:-2].replace(' ', '').split('],[')
                for i in range(len(matrix_p)):
                    _p = list(map(float, matrix_p[i].split(',')))
                    _score = _score + _p[y_true[i]]
                __score = float(_score) * ((len(y_true) - y_true.count(0) + 1) / (y_true.count(0) + 1))
                score.append(__score)
            start = 0
            while True:
                line = mr.readline().strip()
                if not line:
                    break
                l = line.split('\t')
                correct_pep = l[1]
                all_correct_pep.append(correct_pep)
                l = l[2:]
                ##if file mode is missed,append correct peptide at the end
                if file_mode == 'missed':
                    l.append(correct_pep)
                _pep_score = {}
                for one in l:
                    _pep_score[one] = score[start]
                    start += 1
                all_pepscore.append(_pep_score)
            count_len = [0] * 20
            for iii in all_correct_pep:
                count_len[math.ceil(len(iii.split('_')[0]) / 5) - 1] += 1
            print('peptide length : ' + str(count_len))
            total_number = len(pre_index)
            print('total : ' + str(total_number))
            orginal_diss = []
            predict_diss = []
            for c in range(10):
                on = org_index.count(c)
                orate = on / total_number
                pn = pre_index.count(c)
                prate = pn / total_number
                orginal_diss.append(on)
                predict_diss.append(pn)
                print('orginal rank ' + str(c + 1) + ' : ' + str(on) + ' || rate : ' + str(round(orate, 3)))
                print('predict rank ' + str(c + 1) + ' : ' + str(pn) + ' || rate : ' + str(round(prate, 3)))
            orginal_diss.append(total_number - sum(orginal_diss))
            predict_diss.append(total_number - sum(predict_diss))
            print('original : ' + str(orginal_diss))
            print('predict : ' + str(predict_diss))

    '''--------------------------------FDR ROC plot---------------------------------'''
    #   generate all PSMs file and Annotate regular ions
    def get_all_PSMs_and_byions(self):
        comet_results = self.read_comet_results(have_decoy=True,filename=self.workpath+'selected_NCE'+self.nce+'_forcomet.txt')
        correcte_pep, correcte_mgf = self.read_correct_PSMs(filename=self.workpath+'selected_NCE'+self.nce+'.mgf')
        spectrums_number = 0
        index_missed = []
        with open(self.workpath+'FDR/selected_'+self.nce+'_all_PSMs.mgf','a+') as mgfw, open(self.workpath+'FDR/selected_'+self.nce+'_all_PSMs.txt', 'a+') as txtw:
            for _index in tqdm(range(len(correcte_pep))):
                _correcte_seq = correcte_pep[_index]
                try:
                    _comet_seqs = comet_results[str(_index + 1)]
                except:
                    index_missed.append(_index + 1)
                    continue
                flag_index = 1000
                for i in range(len(_comet_seqs)):
                    if _comet_seqs[i].replace(' ', '') == _correcte_seq:
                        flag_index = i
                        break
                if flag_index == 1000:
                    _comet_seqs.append(_correcte_seq)
                spectrums_number += len(_comet_seqs)
                txtw.write(str(_index) + '\t' + _correcte_seq + '\t' + '\t'.join(_comet_seqs) + '\n')
                for i in range(len(_comet_seqs)):
                    o = _comet_seqs[i]
                    if o.startswith('DECOY'):
                        seq = o.split('_')[0].split('-')[1]
                    else:
                        seq = o.split('_')[0]
                    modif = o.split('_')[1]
                    _psm = copy.deepcopy(correcte_mgf[_index])
                    _psm.insert(2, 'Sequence=' + seq + '\n')
                    _psm.insert(4, 'Modified=' + modif + '\n')
                    mgfw.write(''.join(_psm))
        print('total spectrums number : ' + str(spectrums_number))
        print(index_missed)
        m = MATCH(self.workpath+'FDR', 'selected_'+self.nce+'_all_PSMs.mgf')
        m.write_files()

    #   split Annotated files for P-score,Because it takes up too much memory
    def split_byions(self,each_number=100000):
        with open(self.workpath+'FDR/selected_'+self.nce+'_all_PSMs_byions.txt','r') as r:
            count = 0
            while True:
                line = r.readline()
                if not line.strip():
                    break
                pep_length = len(line.split('\t')[0])
                _line = []
                _line.append(line)
                for i in range(pep_length - 2):
                    line = r.readline()
                    _line.append(line)
                _flag = int(count / each_number) + 1
                with open(self.workpath+'FDR/splited_by_ions/all_psms_spectrums_byions' + str(_flag) + '.txt', 'a+') as w:
                    w.write(''.join(_line))
                _line = []
                count += 1
            print(count)

    #   Get FDR ROC plot Data file of P-score
    def get_pscore_FDR(self, split_by_charge=False):
        split_CHARGE = []
        all_charge = []
        Length = []
        with open(self.workpath+'FDR/selected_'+self.nce+'_all_PSMs.mgf', 'r') as r:
            line = r.readline()
            last_title = ''
            start = 0
            while True:
                if not line.strip():
                    break
                if line.startswith('TITLE='):
                    _title = line.strip().split('=')[1]
                if line.startswith('CHARGE='):
                    _charge = line.strip().split('=')[1]
                    all_charge.append(int(_charge))
                    if last_title != _title:
                        split_CHARGE.append(_charge)
                        last_title = _title
                        Length.append(start)
                        start = 0
                    start += 1
                line = r.readline()
        print(len(split_CHARGE))
        print(len(all_charge))
        print(len(Length))
        candidate_peps = []
        correcte_peps = []
        with open(self.workpath+'FDR/selected_'+self.nce+'_all_PSMs.txt', 'r') as r:
            line = r.readline()
            while True:
                if not line.strip():
                    break
                _candidate = line.strip().split('\t')[2:]
                _correcte = line.strip().split('\t')[1]
                candidate_peps.append(_candidate)
                correcte_peps.append(_correcte)
                line = r.readline()
            print(len(candidate_peps))
        with open(self.workpath+'FDR/selected_'+self.nce+'_score_pep_P_4label_humanmodel.txt','r') as r:
            line = r.readline()
            score = []
            Y = []
            l1 = ''
            l2 = ''
            while True:
                if not line.strip():
                    print('read score end!')
                    break
                l1 = l2
                l2 = line
                line = line.strip().split('\t')
                try:
                    y_true = list(map(int, line[0].split(',')))
                except:
                    print(l1)
                    print(l2)
                y_pred = list(map(int, line[1].split(',')))
                Y.append([line[0], line[1]])
                _score = 1.0
                matrix_p = line[8].strip()[2:-2].replace(' ', '').split('],[')
                for i in range(len(matrix_p)):
                    _p = list(map(float, matrix_p[i].split(',')))
                    _score = _score + _p[y_true[i]]
                _score = float(_score) * ((len(y_true) - y_true.count(0) + 1) / (y_true.count(0) + 1))
                score.append(_score)
                line = r.readline()
        print(len(score))
        start = 0
        top_pep_charge_score = []
        threshold_score = []
        for i in range(len(candidate_peps)):
            _candidate = candidate_peps[i]
            _pep_charge_score = []
            _charge = split_CHARGE[i]
            _correcte = correcte_peps[i]
            for one in _candidate:
                _score = score[start]
                _y = Y[start]
                _pep_charge_score.append([one, _charge, _score] + _y + [_correcte])
                start += 1
            _c = [x for x in _pep_charge_score if x[0] == x[5]]
            _pep_charge_score = sorted(_pep_charge_score, key=lambda x: x[2], reverse=True)
            top_pep_charge_score.append(_pep_charge_score[0] + _c[0])
            t = _pep_charge_score[0][2]
            if t not in threshold_score:
                threshold_score.append(t)
        threshold_score = sorted(threshold_score)
        print(threshold_score)
        # write top1 ,format:   pep charge score y_true y_pred _correcte
        with open(
                self.workpath+'FDR/results/Decoy_score_P_4label_humanmodel_allcharge_changescore.txt',
                'a+') as w, open(
                self.workpath+'FDR/results/missed_P_4label_humanmodel_allcharge_changescore.txt',
                'a+') as mw:
            for i in top_pep_charge_score:
                _line = '\t'.join(list(map(str, i))) + '\n'
                w.write(_line)
                if i[0] != i[5]:
                    __line = '\t'.join(list(map(str, i))) + '\n'
                    mw.write(__line)
            print('write top 1 end !')
        if split_by_charge:
        ## get FDR by splite charge
            with open(self.workpath+'FDR/results/FDR_results_P_4label_humanmodel.txt', 'a+') as w:
                for _index in tqdm(range(len(threshold_score))):
                    t = threshold_score[_index]
                    target_hits = [0,0,0,0,0]
                    threshold_all = list(x for x in top_pep_charge_score if x[2]>=t)
                    for o in threshold_all:
                        if o[0] == o[5]:
                            target_hits[int(o[1])-2] += 1
                    for c in ['2','3','4','5']:
                        threshold_seq_score = list(x for x in threshold_all if x[1]==c)
                        False_seq_score = list(x for x in threshold_seq_score if x[0].startswith('DECOY-'))
                        try:
                            False_Discover_Rate = len(False_seq_score) / (len(threshold_seq_score)-len(False_seq_score))
                        except:
                            False_Discover_Rate = 0.0
                        _line = 'Threshold peptide score : ' + str(t) + '\thold number : ' + str(len(threshold_seq_score)) + '\tFDR : ' + str(False_Discover_Rate) + '\ttarget hits : ' + str(target_hits[int(c)-2]) + '\tcharge : ' + str(c)
                        w.write(_line + '\n')
        else:
        ## get FDR don't splite charge
            with open(
                    self.workpath+'FDR/results/FDR_results_P_4label_humanmodel_allcharge_changescore.txt',
                    'a+') as w:
                for _index in tqdm(range(len(threshold_score))):
                    t = threshold_score[_index]
                    target_hits = 0
                    threshold_all = list(x for x in top_pep_charge_score if x[2] >= t)
                    for o in threshold_all:
                        if o[0] == o[5]:
                            target_hits += 1
                    False_seq_score = list(x for x in threshold_all if x[0].startswith('DECOY-'))
                    try:
                        False_Discover_Rate = len(False_seq_score) / (len(threshold_all) - len(False_seq_score))
                    except:
                        False_Discover_Rate = 0.0
                    _line = 'Threshold peptide score : ' + str(t) + '\thold number : ' + str(
                        len(threshold_all)) + '\tFDR : ' + str(False_Discover_Rate) + '\ttarget hits : ' + str(target_hits)
                    w.write(_line + '\n')

    #   Get FDR ROC plot Data file of Comet
    def get_comet_FDR(self, split_by_charge=False):
        score_type = 0      ##0 is xcorr,1 is evalue
        split_CHARGE = []
        all_charge = []
        with open(self.workpath+'selected_NCE'+self.nce+'.mgf', 'r') as r:
            line = r.readline()
            last_title = ''
            while True:
                if not line.strip():
                    break
                if line.startswith('TITLE='):
                    _title = line.strip().split('=')[1]
                if line.startswith('CHARGE='):
                    _charge = line.strip().split('=')[1]
                    all_charge.append(int(_charge))
                    if last_title != _title:
                        split_CHARGE.append(_charge)
                        last_title = _title
                line = r.readline()
        print(len(split_CHARGE))
        print(len(all_charge))
        comet_results = self.read_comet_results(have_decoy=True, have_score=score_type,filename=self.workpath+'selected_NCE'+self.nce+'_forcomet.txt')
        print(len(comet_results))
        correcte_results, correcte_spectrum = self.read_correct_PSMs(filename=self.workpath+'selected_NCE'+self.nce+'.mgf')
        print(len(correcte_results))
        threshold_score = []
        for k, v in comet_results.items():
            _correcte = correcte_results[int(k) - 1]
            _charge = split_CHARGE[int(k) - 1]
            _v = v[0]
            _v.extend([_charge, _correcte])  ##[pep,xcorr,charge,correcte]
            comet_results[k] = _v
            t = round(float(v[0][1]), 4)
            if t not in threshold_score:
                threshold_score.append(t)
        if score_type == 0:
            threshold_score = sorted(threshold_score, reverse=False)  ##Xcorr:False;E-value:True
        elif score_type == 1:
            threshold_score = sorted(threshold_score, reverse=True)
        print(threshold_score)
        print(comet_results)
        # write top1 ,format:   pep xcorr charge correct_pep
        with open(
                self.workpath+'FDR/results/comet_Decoy_score_xcorr_allcharge.txt',
                'a+') as w, open(
                self.workpath+'FDR/results/comet_xcorr_missed_allcharge.txt',
                'a+') as mw:
            for key, value in comet_results.items():
                _line = '\t'.join(value) + '\n'
                w.write(_line)
                if value[0] != value[3]:
                    mw.write(_line)
        ## get FDR by splite charge
        if split_by_charge:
            with open(self.workpath+'FDR/results/comet_FDR_results_xcorr.txt','a+') as w:
                for t in tqdm(threshold_score):
                    target_hits = [0, 0, 0, 0, 0]  # charge:2,3,4,5
                    FDR_count = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
                    threshold_seq_score = list(
                        (key, value) for key, value in comet_results.items() if float(value[1]) >= t)  ##Xcorr:>;E-value:<
                    False_seq_score = list(
                        (key, value) for key, value in threshold_seq_score if value[0].startswith('DECOY-'))
                    _c = list((key, value) for key, value in threshold_seq_score if value[0] == value[3])
                    for i in range(len(target_hits)):
                        FDR_count[i][0] += len(
                            list((key, value) for key, value in False_seq_score if int(value[2]) == (i + 2)))
                        FDR_count[i][1] += len(
                            list((key, value) for key, value in threshold_seq_score if int(value[2]) == (i + 2)))
                        target_hits[i] = len(list((key, value) for key, value in _c if int(value[2]) == (i + 2)))
                    for i in [2, 3, 4, 5]:
                        try:
                            False_Discover_Rate = FDR_count[i - 2][0] / (FDR_count[i - 2][1] - FDR_count[i - 2][0])
                        except:
                            False_Discover_Rate = 0.0
                        _line = 'Threshold peptide score : ' + str(t) + '\thold number : ' + str(
                            FDR_count[i - 2][1]) + '\tFDR : ' + str(False_Discover_Rate) + '\ttarget hits : ' + str(
                            target_hits[i - 2]) + '\tcharge : ' + str(i)
                        # print(_line)
                        w.write(_line + '\n')
        else:
        ## get FDR don't splite charge
            with open(self.workpath+'FDR/results/comet_FDR_results_xcorr_allcharge.txt','a+') as w:
                for t in tqdm(threshold_score):
                    target_hits = 0
                    FDR_count = [0,0]
                    threshold_seq_score = list((key,value) for key,value in comet_results.items() if float(value[1]) >= t)      ##Xcorr:>;E-value:<
                    False_seq_score = list((key,value) for key,value in threshold_seq_score if value[0].startswith('DECOY-'))
                    _c = list((key,value) for key,value in threshold_seq_score if value[0] == value[3])
                    FDR_count[0] += len(list((key,value) for key,value in False_seq_score))
                    FDR_count[1] += len(list((key,value) for key,value in threshold_seq_score))
                    target_hits = len(list((key,value) for key,value in _c))
                    try:
                        False_Discover_Rate = FDR_count[0]/(FDR_count[1]-FDR_count[0])
                    except:
                        False_Discover_Rate = 0.0
                    _line = 'Threshold peptide score : ' + str(t) + '\thold number : '+ str(FDR_count[1]) + '\tFDR : ' + str(False_Discover_Rate) + '\ttarget hits : ' + str(target_hits)
                    w.write(_line+'\n')

if __name__ == '__main__':
    proteometools = ProteomeTools(workpath='E:/data/1/get_ions/ProteomeTools2/selected_mgf2',nce='30')
    proteometools.find_unkonwn_aa()
    ##top1 hits rate
    proteometools.get_different_peptide()
    proteometools.get_byions()
    proteometools.get_MatrixP()
    proteometools.eval_prediction()
    ##FDR ROC plot
    proteometools.get_all_PSMs_and_byions()
    proteometools.split_byions()
    proteometools.get_pscore_FDR()
    proteometools.get_comet_FDR()