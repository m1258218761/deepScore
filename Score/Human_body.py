#coding=utf-8
import os
import numpy as np
import copy
from tqdm import tqdm
from  data_util import data
from match_ions import MATCH
import torch
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from Resnet_model import ResNet18

class Humanbody_proteome(object):
    def __init__(self,workpath='',nce=''):
        self.workpath = workpath
        self.nce = nce
        self.top1_workpath = self.workpath+'/'+self.nce+'_random10000/pep_credibility/'
        self.fdr_workpath = self.workpath+'/'+self.nce+'_fdr/'

    '''
     stage 1 : Preparation before Score by Comet and P-score,including preprocessing of spectrum,Atfer stage 1,Before
     the stage 2 begins, you need to use Comet for identification
    '''
    #---Basic function---:reade the original .mgf file
    def read_mgf(self,file,file_path):
        mgf_content ={}
        mgf_listcontent = []
        with open(file_path + '/' + file, 'r') as rf:
            while True:
                line = rf.readline()
                if not line:
                    break
                _line = ''
                if 'BEGIN IONS' in line:
                    title = ''
                    while True:
                        _line += line
                        line = rf.readline()
                        if 'TITLE=' in line:
                            title = line.split('=')[1].replace('\n','')
                        if 'END IONS' in line:
                            _line += line
                            break
                    mgf_listcontent.append(_line)
                    mgf_content[title] = _line
        return mgf_content,mgf_listcontent

    #   Generate .mgf files corresponding to train,validation and test data
    def get_mgf(self,mgf_path,data_path):
        mgf_list = os.listdir(mgf_path)
        # test_list = ['30_test_by.txt','35_test_by.txt','40_test_by.txt']
        test_list = ['30_train_by.txt','35_train_by.txt','40_train_by.txt','30_validation_by.txt','35_validation_by.txt','40_validation_by.txt']
        mgf_content = {}
        for mgf in mgf_list:
            extention = os.path.splitext(mgf)
            if extention[1] == '.mgf':
                mgf_content[extention[0]],_ = self.read_mgf(mgf,mgf_path)
                print('[Preparation Info]read mgf : ' + str(mgf))
        for file in test_list:
            with open(data_path + '/' + file,'r') as r,open(data_path + '/pep_credibility/' + file[:-4] + '.mgf','a+') as w:
                content = []
                line = r.readline()
                while True:
                    if not line.strip('\n'):
                        break
                    pairs_count = len(line.split('\t')[0]) - 2
                    i = 0
                    _line = line.split('\t')
                    peptide = _line[0]
                    title =_line[3]
                    content.append([title,peptide])
                    while i <= pairs_count and line.strip('\n'):
                        line = r.readline()
                        i += 1
                    p_number = title.split('_')[2]
                    w.write(mgf_content[p_number][title])
            print('[Preparation Info]write : ' + file)

    #   Select 10000 spectrum for Assessment of top1 hit rate
    def choice_scpctrum(self):
        _,mgf_listcontent = self.read_mgf(self.nce+'_test_by.mgf',self.workpath+'/mgf')
        _index = np.random.choice(len(mgf_listcontent),10000,replace=False).tolist()
        print('[Preparation Info]selected index : ' + str(_index))
        print('[Preparation Info]selected total number : ' + str(len(_index)))
        with open(self.top1_workpath+'selected_'+self.nce+'.mgf','a+') as mw:
            for i in _index:
                mw.write(mgf_listcontent[i])
            print('[Preparation Info]write selected spectrum success ! ')

    #   Remove additional information from the spectrum to generate a spectrum that can be identified by Comet
    def get_search_spectrumforcomet(self):
        with open(self.top1_workpath+'selected_'+self.nce+'.mgf', 'r') as rf, open(self.top1_workpath + 'selected_'+self.nce+'_forcomet.mgf', 'a+') as w:
            while True:
                line = rf.readline()
                if not line:
                    break
                _line = ''
                if 'BEGIN IONS' in line:
                    title = ''
                    _line += line
                    while True:
                        line = rf.readline()
                        if 'Sequence=' in line or 'Modified=' in line or 'Collison Energy=' in line or 'PEP=' in line:
                            continue
                        if 'CHARGE=' in line:
                            line = line.strip() + '+\n'
                        _line += line
                        if 'END IONS' in line:
                            break
                w.write(_line)

    #   Merge Protein Sequence Database of Humanbody proteome for Comet
    def get_fasta(self,workpath=''):
        with open(workpath + '/crap.fasta','r') as r1,open(workpath + '/uniprot-proteome-UP000005640-20190603.fasta','r')as r2,open(workpath + '/Human.fasta','a+') as w:
            one_seq = ''
            flag = 0
            for line in r1:
                if line.startswith('>'):
                    if flag == 1:
                        w.write(one_seq)
                    name = line.replace('>', '').split()[0]
                    if 'HUMAN' in name:
                        flag = 1
                    else:
                        flag = 0
                    one_seq = ''
                one_seq += line

    '''
     stage 2 : Scoring with P-score ,then Compare Comet and P-score in ROC plot with random selected .raw and top1
     hits rate with random selected 10000 spectrums
     '''
     #  Delete the unconventional amino acids from Comet identification results:U
    def find_unkonwn_aa(self):
        ## random 10000
        with open(self.top1_workpath+'selected_'+self.nce+'_forcomet.txt','r') as r, open(
                self.top1_workpath+'_selected_'+self.nce+'_forcomet.txt', 'a+') as w:
            ## random raw
            # with open('E:/data/1/get_ions/by_ions_PPM/40_fdr/00603_A02_P004608_B00I_A00_R1_HCDFT.txt','r') as r,open('E:/data/1/get_ions/by_ions_PPM/40_fdr/_00603_A02_P004608_B00I_A00_R1_HCDFT.txt','a+') as w:
            r.__next__()
            r.__next__()
            while True:
                line = r.readline()
                if not line.strip():
                    break
                l = line.split('\t')
                if 'U' in l[11]:
                    print('[Preparation Info]deleted line : ' + str(line))
                else:
                    w.write(line)


    ###---Basic function---:read identification results and return
    ###   Parameter: have_decoyt:Return results include Decoy;
    ###              have_score:0 means return score, 1 means return evalue;
    ###              have_charge:retrun peptide charge
    ###              filename:identification file
    #   Comet
    def read_comet_results(self,have_decoy=False, have_score=0, have_charge=False, filename=''):
        with open(filename, 'r') as rf:
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
        print('[Score Info]Comet results number : ' + str(len(results)))
        if have_charge:
            return results, CHARGE
        else:
            return results

    #   MSGF+
    def read_msgf_results(self,have_decoy=False, have_score=0, have_charge=False, filename=''):
        with open('E:/data/1/get_ions/by_ions_PPM/30_fdr/MSGF/00705_F03_P005217_B0V_A00_R2_HCDFT.tsv', 'r') as rf:
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
                _score = l[12]
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
            print('[Score Info]MSGF+ results number : ' + str(len(_results)))
            if have_charge:
                return _results, CHARGE
            else:
                return _results

    #---Basic function---:get correct peptide and spectrum
    def read_correct_PSM(self, filename=''):
        with open(filename, 'r') as rf:
            mgf_listcontent = []
            correct_peptide = []
            while True:
                line = rf.readline()
                if not line:
                    break
                _line = []
                if 'BEGIN IONS' in line:
                    _line.append(line)
                    while True:
                        line = rf.readline()
                        if 'Sequence=' in line:
                            _seq = line.strip().split('=')[1]
                            _temp = _seq
                        if 'Modified=' in line:
                            _modeified = line.split('=')[1].strip().replace(' ', '').split(';')
                            _C = [_modeified[-1]]
                            if _modeified[:-1] != ['']:
                                _M = sorted(_modeified[:-1], key=lambda x: int(x.split('@M')[1]))
                            else:
                                _M = _modeified[:-1]
                            _modeified = _M + _C
                            _temp += '_' + ';'.join(_modeified)
                            correct_peptide.append(_temp)
                        if 'Sequence=' in line or 'Modified=' in line or 'Collison Energy=' in line or 'PEP=' in line or 'PIF=' in line:
                            continue
                        else:
                            _line.append(line)
                        if 'END IONS' in line:
                            mgf_listcontent.append(_line)
                            _line = []
                            break
        print('[Score Info]correct results number : ' + str(len(correct_peptide)))
        return correct_peptide, mgf_listcontent

    '''-------------------------------top1 hit rate(random 10000 spectrum)-----------------------------'''
    #   Evaluation of comet identification results and generate related files,include Comet top1 missed and unmissed
    def get_different(self):
        total_PSMs = 0
        count = 0
        unmissed_total_PSMs = 0
        unmissed_count = 0
        with open(self.top1_workpath+'selected_'+self.nce+'_missed_peptide.txt','a+') as mtw, open(self.top1_workpath+'selected_'+self.nce+'_missed_PSMs.mgf',
                'a+') as mgw, open(self.top1_workpath+'selected_'+self.nce+'_unmissed_PSMs.mgf','a+') as ugw,open(self.top1_workpath+'selected_'+self.nce+'_unmissed_peptide.txt','a+') as utw:
            comet_results, CHARGE = self.read_comet_results(have_decoy=False, have_charge=True,filename=self.top1_workpath+'selected_30_forcomet.txt')
            correcte_results, correcte_spectrum = self.read_correct_PSM(filename=self.top1_workpath+'selected_'+self.nce+'.mgf')
            for i in range(len(correcte_results)):
                correcte_seq = correcte_results[i]
                if comet_results.get(str(i + 1)) == None:
                    print('[Score Info]comet have no peptide index : ' + str(i + 1))
                    continue
                comet_seq = comet_results[str(i + 1)]
                c_index = 1000
                for index in range(len(comet_seq)):
                    if comet_seq[index].replace(' ', '') == correcte_seq:
                        c_index = index
                        break
                if c_index != 0:
                    _charge = CHARGE[str(i + 1)]
                    mtw.write(str(i) + '\t' + correcte_seq + '\t' + '\t'.join(comet_seq) + '\t' + str(_charge) + '\n')
                    print(correcte_seq)
                    print(comet_seq)
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
                    _charge = CHARGE[str(i + 1)]
                    utw.write(str(i) + '\t' + correcte_seq + '\t' + '\t'.join(comet_seq) + '\t' + str(_charge) + '\n')
                    unmissed_count += 1
                    unmissed_total_PSMs += len(comet_seq)
                    for o in comet_seq:
                        seq = o.split('_')[0]
                        modif = o.split('_')[1]
                        _psm = copy.deepcopy(correcte_spectrum[i])
                        _psm.insert(2, 'Sequence=' + seq + '\n')
                        _psm.insert(4, 'Modified=' + modif + '\n')
                        ugw.write(''.join(_psm))
            print('[Score Info]missed peptide number : ' + str(count))
            print('[Score Info]missed total PSMs : ' + str(total_PSMs))
            print('[Score Info]unmissed peptide number : ' + str(unmissed_count))
            print('[Score Info]unmissed total PSMs : ' + str(unmissed_total_PSMs))

    #   Annotate regular ions(b1+,y1+,b2+,y2+) and generate the files can be scored by P-score
    def get_byions(self):
        m = MATCH(self.top1_workpath, 'selected_'+self.nce+'_missed_PSMs.mgf')
        m.write_files()
        um = MATCH(self.top1_workpath, 'selected_'+self.nce+'_unmissed_PSMs.mgf')
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
        Data = data(self.top1_workpath, Label_number,run_model='Test',test_file=file)
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
            with open( self.top1_workpath+'sorted_by_pccandother/' + file_mode + '_score_pep_P.txt','a+') as fw:
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
        file_mode = 'missed'
        all_pepscore = []
        all_correct_pep = []
        all_charge = []
        pre_index = []
        pre_index_bycharge = [[] for i in range(5)]
        org_index = []
        org_index_bycharge = [[] for i in range(5)]
        with open(
                                self.workpath+'/30_random10000/pep_credibility/selected_30_' + file_mode + '_peptide.txt',
                                'r') as mr, open(
                                self.workpath+'/30_random10000/pep_credibility_4label/sorted_by_pccandother/' + file_mode + '_score_pep_P.txt',
                                'r') as fr:
            score = []
            print('[Score Info]start reading score...')
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
            print(len(score))
            start = 0
            while True:
                line = mr.readline().strip()
                if not line:
                    break
                l = line.split('\t')
                correct_pep = l[1]
                all_correct_pep.append(correct_pep)
                _charge = int(l[-1]) - 2
                all_charge.append(_charge)
                l = l[2:-1]
                ##if file mode is missed,append correct peptide at the end
                if file_mode == 'missed':
                    l.append(correct_pep)
                _pep_score = {}
                for one in l:
                    _pep_score[one] = score[start]
                    start += 1
                all_pepscore.append(_pep_score)
            print(len(all_pepscore[0]) + len(all_pepscore[1]) + len(all_pepscore[2]))
            print(len(all_pepscore[3]))
            for _i in tqdm(range(len(all_pepscore))):
                o_index = np.where(np.array(list(all_pepscore[_i].items()))[:, 0] == all_correct_pep[_i])[0][0]
                org_index.append(o_index)
                org_index_bycharge[int(all_charge[_i])].append(o_index)
                _pep = np.array(sorted(all_pepscore[_i].items(), key=lambda x: float(x[1]), reverse=True))
                _index = np.where(_pep[:, 0] == all_correct_pep[_i])[0][0]
                pre_index.append(_index)
                pre_index_bycharge[int(all_charge[_i])].append(_index)
            total_number = len(pre_index)
            _total_number = []
            for i in range(len(pre_index_bycharge)):
                _total_number.append(len(pre_index_bycharge[i]))
            print('total : ' + str(total_number))
            orginal_diss = []
            predict_diss = []
            orginal_diss_charge = [[] for i in range(5)]
            predict_diss_charge = [[] for i in range(5)]
            for c in range(10):
                on = org_index.count(c)
                orate = on / total_number
                pn = pre_index.count(c)
                prate = pn / total_number
                orginal_diss.append(on)
                predict_diss.append(pn)
                for i in range(len(org_index_bycharge)):
                    orginal_diss_charge[i].append(org_index_bycharge[i].count(c))
                    predict_diss_charge[i].append(pre_index_bycharge[i].count(c))
                print('[Score Info]orginal rank ' + str(c + 1) + ' : ' + str(on) + ' || rate : ' + str(round(orate, 3)))
                print('[Score Info]predict rank ' + str(c + 1) + ' : ' + str(pn) + ' || rate : ' + str(round(prate, 3)))
            orginal_diss.append(total_number - sum(orginal_diss))
            predict_diss.append(total_number - sum(predict_diss))
            for i in range(len(orginal_diss_charge)):
                orginal_diss_charge[i].append(_total_number[i] - sum(orginal_diss_charge[i]))
                predict_diss_charge[i].append(_total_number[i] - sum(predict_diss_charge[i]))
            print('[Score Info]original : ' + str(orginal_diss))
            print('[Score Info]predict : ' + str(predict_diss))
            for i in range(len(orginal_diss_charge)):
                print('[Score Info]charge ' + str(i + 2) + ' original: ' + str(orginal_diss_charge[i]))
                print('[Score Info]charge ' + str(i + 2) + ' predict: ' + str(predict_diss_charge[i]))

    '''---------------------------------FDR ROC plot(random .raw file)---------------------------------'''
    #   Get FDR ROC plot Data file of Search engine
    def get_search_fdr(self, split_by_charge=False):
        score_type = 0      ##0 is xcorr,1 is evalue
        comet_results, CHARGE = self.read_comet_results(have_decoy=True, have_score=score_type, have_charge=True,filename=self.fdr_workpath+'00705_F03_P005217_B0V_A00_R2_HCDFT.txt')
        keys = list(comet_results.keys())
        keys = sorted(keys, key=lambda x: int(x))  # if MSGF+
        top_pep_xcorr = []
        for key in keys:
            _pep_xcorr = comet_results[key][0]
            _pep_xcorr.extend(CHARGE[key])
            top_pep_xcorr.append(_pep_xcorr)
        print('[Score Info]comet top1 number : ' + str(len(top_pep_xcorr)))
        score_threshold = []
        ## get Comet FDR by splite charge
        if split_by_charge == True:
            with open(self.fdr_workpath+'comet_top1_pep_xcorr.txt', 'a+') as w, open(self.fdr_workpath+'comet_FDR_results_xcorr.txt', 'a+') as ww:
                for i in range(len(top_pep_xcorr)):
                    _line = '\t'.join(top_pep_xcorr[i]) + '\n'
                    w.write(_line)
                    _xcorr = top_pep_xcorr[i][1]
                    if _xcorr not in score_threshold:
                        score_threshold.append(float(_xcorr))
                score_threshold = sorted(score_threshold, reverse=False)  ##Xcorr:False;E-value:True
                for t in tqdm(score_threshold):
                    threshold_pep = list(x for x in top_pep_xcorr if float(x[1]) >= t)  ##Xcorr:>;E-value:<
                    for c in ['2', '3', '4', '5', '6']:
                        c_pep = list(x for x in threshold_pep if x[2] == c)
                        f_pep = list(x for x in c_pep if x[0].startswith('DECOY-'))
                        decoy = len(f_pep)
                        target = len(c_pep) - decoy
                        try:
                            False_Discover_Rate = decoy / target
                        except:
                            False_Discover_Rate = 0.0
                        _line = 'Threshold peptide score : ' + str(t) + '\ttarget number : ' + str(
                            target) + '\tFDR : ' + str(False_Discover_Rate) + '\tcharge : ' + str(c) + '\n'
                        ww.write(_line)
        ## get Comet FDR not splite charge
        elif split_by_charge == False:
            with open(self.fdr_workpath+'comet_top1_pep_xcorr_allcharge.txt','a+') as w,open(self.fdr_workpath+'comet_FDR_results_xcorr_allcharge.txt','a+') as ww:
                for i in range(len(top_pep_xcorr)):
                    _line = '\t'.join(top_pep_xcorr[i]) + '\n'
                    w.write(_line)
                    _xcorr = top_pep_xcorr[i][1]
                    if _xcorr not in score_threshold:
                        score_threshold.append(float(_xcorr))
                if score_type == 0:
                    score_threshold = sorted(score_threshold, reverse=False)     ##Xcorr:False;E-value:True
                elif score_type == 1:
                    score_threshold = sorted(score_threshold, reverse=True)
                for t in tqdm(score_threshold):
                    threshold_pep = list(x for x in top_pep_xcorr if float(x[1])>= t)   ##Xcorr:>;E-value:<
                    f_pep = list(x for x in threshold_pep if x[0].startswith('DECOY-'))
                    decoy = len(f_pep)
                    target = len(threshold_pep)-decoy
                    try:
                        False_Discover_Rate = decoy/target
                    except:
                        False_Discover_Rate = 0.0
                    _line = 'Threshold peptide score : ' + str(t) + '\ttarget number : '+ str(target) + '\tFDR : ' + str(False_Discover_Rate) +'\n'
                    ww.write(_line)

    #   generate all PSMs file and Annotate regular ions
    def get_all_psms_and_byions(self):
        comet_results = self.read_comet_results(have_decoy=True,filename=self.fdr_workpath+'00705_F03_P005217_B0V_A00_R2_HCDFT.txt')
        _comet_index = list(comet_results.keys())
        _comet_index = sorted(_comet_index, key=lambda x: int(x))  # if MSGF+
        ## get spectrum
        with open(self.fdr_workpath+'00631_E02_P004778_B00M_A00_R1_CIDFT.mgf', 'r') as rf:
            _spectrum_content = []
            start = 1
            while True:
                line = rf.readline()
                if not line:
                    break
                _line = []
                if 'BEGIN IONS' in line:
                    _line.append(line)
                    while True:
                        line = rf.readline()
                        _line.append(line)
                        if 'END IONS' in line:
                            if str(start) in _comet_index:
                                _spectrum_content.append(_line)
                            start += 1
                            _line = []
                            break
        print('[Score Info]spectrum number : ' + str(len(_spectrum_content)))
        ##product all psms spectrum
        with open(self.fdr_workpath+'all_psms_spectrums.mgf', 'a+') as w:
            for _index in tqdm(range(len(_spectrum_content))):
                _spectrum = _spectrum_content[_index]
                __index = _comet_index[_index]
                _candidate_pep = comet_results[__index]
                for _candidate in _candidate_pep:
                    _line = []
                    if _candidate.startswith('DECOY-'):
                        _pep = _candidate.split('-')[1].split('_')[0]
                    else:
                        _pep = _candidate.split('_')[0]
                    _Modified = _candidate.split('_')[1]
                    for _l in _spectrum:
                        if _l.startswith('CHARGE='):
                            _l = _l.replace('+', '')
                            _line.append('Sequence=' + _pep + '\n')
                        if _l.startswith('RTINSECONDS='):
                            _line.append('Modified=' + _Modified + '\n')
                        if _l.startswith('PIF='):
                            continue
                        _line.append(_l)
                    w.write(''.join(_line))
        m = MATCH(self.fdr_workpath, 'all_psms_spectrums.mgf')
        m.write_files()

    #   split Annotated files for P-score,Because it takes up too much memory
    def split_byions(self, each_number=100000):
        with open(self.fdr_workpath+'all_psms_spectrums_byions.txt', 'r') as r:
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
                with open(self.fdr_workpath+'splited_by_ions/all_psms_spectrums_byions' + str(_flag) + '.txt', 'a+') as w:
                    w.write(''.join(_line))
                _line = []
                count += 1
            print(count)

    #   Get FDR ROC plot Data file of P-score
    def get_pscore_fdr(self, split_by_charge=False):
        pep_score = []
        with open(self.fdr_workpath+self.nce+'_fdr_pep_score_4label_P.txt', 'r') as sr:
            while True:
                line = sr.readline()
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
                _score = float(_score) * ((len(y_true) - y_true.count(0) + 1) / (y_true.count(0) + 1))
                pep_score.append([line[0], line[1], str(_score)])  ##y_true,y_pred,score
        print('[Score Info]score number : ' + str(len(pep_score)))
        top_score_pep = []
        comet_results, CHARGE = self.read_comet_results(have_decoy=True, have_charge=True)
        keys = list(comet_results.keys())
        keys = sorted(keys, key=lambda x: int(x))  # if MSGF+
        start = 0
        for key in keys:
            one = []
            for a in comet_results[key]:
                a = [a, CHARGE[key]]
                a = a + pep_score[start]
                one.append(a)
                start += 1
            sorted_one = sorted(one, key=lambda x: float(x[4]), reverse=True)
            top_score_pep.append(sorted_one[0])
        print('[Score Info]top1 number : ' + str(len(top_score_pep)))
        ## get score threshold and write top1 txt
        score_threshold = []
        with open(self.fdr_workpath+'top1_pep_4label_P_changescore_allcharge.txt', 'a+') as aw:
            for l in top_score_pep:
                _score = float(l[4])
                if _score not in score_threshold:
                    score_threshold.append(_score)
                _line = '\t'.join(l) + '\n'
                aw.write(_line)
            print('[Score Info]write top1 peptide and score end !')
        score_threshold = sorted(score_threshold)
        print(score_threshold)
        print('[Score Info]score threshold number : ' + str(len(score_threshold)))
        ## get FDR split charge
        if split_by_charge == True:
            with open(self.fdr_workpath+'FDR_results_4label_P_changescore.txt','a+') as fw:
                for t in tqdm(score_threshold):
                    threshold_pep = list([x for x in top_score_pep if float(x[4])>=t])
                    for c in ['2','3','4','5','6']:
                        threshold_charge = list([x for x in threshold_pep if x[1] == c])
                        false_pep = list([x for x in threshold_charge if x[0].startswith('DECOY-')])
                        false = len(false_pep)
                        target = len(threshold_charge)-false
                        try:
                            False_Discover_Rate = false/target
                        except:
                            False_Discover_Rate = 0.0
                        _line = 'Threshold peptide score : ' + str(t) + '\ttarget number : '+ str(target) + '\tFDR : ' + str(False_Discover_Rate) +'\tcharge : ' + str(c)+'\n'
                        fw.write(_line)
        ## get FDR don't split charge
        elif split_by_charge == False:
            with open(self.fdr_workpath+'FDR_results_4label_P_changescore_allcharge.txt', 'a+') as fw:
                for t in tqdm(score_threshold):
                    threshold_pep = list([x for x in top_score_pep if float(x[4]) >= t])
                    false_pep = list([x for x in threshold_pep if x[0].startswith('DECOY-')])
                    false = len(false_pep)
                    target = len(threshold_pep) - false
                    try:
                        False_Discover_Rate = false / target
                    except:
                        False_Discover_Rate = 0.0
                    _line = 'Threshold peptide score : ' + str(t) + '\ttarget number : ' + str(target) + '\tFDR : ' + str(
                        False_Discover_Rate) + '\n'
                    fw.write(_line)

if __name__ == '__main__':
    human = Humanbody_proteome(workpath='E:/data/1/get_ions/by_ions_PPM',nce='30')
    human.find_unkonwn_aa()
    ##random selected 10000 spectrum for top1 hits rate
    human.get_different()
    human.get_byions()
    human.get_MatrixP()
    human.eval_prediction()
    ##random selected .raw file for FDR ROC plot
    human.get_search_fdr()
    human.get_all_psms_and_byions()
    human.split_byions()
    human.get_MatrixP()
    human.get_pscore_fdr()