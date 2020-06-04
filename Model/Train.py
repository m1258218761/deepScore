# coding=utf-8

import os
import gc
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import classification_report

from Model.config_reader import MyParser
from Model.data_util import data
from Model.Resnet_model import ResNet18
from Model.BiLstm_CRF_model import BiLstm_CRF


# NCE = '30'  ##NCE = 30 or 35
# run_model = 'Train'  ## run model:Train or Test
#
# ##Model parameters
# BATCH_SIZE = 16  ##batch size
# Label_number = 4  ##label 2:predict b1+,y1+;   label 4:predict b1+,b2+,y1+,y2+
# features_size = 105  ##feature size of every Peptide bond
#
# #   label weight for cross entropy
# weights4 = [0.6550, 0.1455, 0.0718, 0.0385, 0.0236, 0.0158, 0.0113, 0.0084, 0.0063, 0.0047, 0.0190]


class Pscore_Model(object):
    def __init__(self):
        conf_path = os.path.dirname(os.path.realpath(__file__)) + "/config.ini"
        cfg = MyParser()
        cfg.read(conf_path, encoding='utf-8')
        parameters = cfg.as_dict()
        tp = parameters['Model_Train']
        self.NCE = tp['nce']
        self.epoch = int(tp['epoch'])
        self.lr_r = float(tp['r'])
        self.lr_rate = float(tp['lr_rate'])
        self.run_model = tp['run_model']
        self.pretrained = tp['pretrained']
        self.hold_threshold = int(tp['h_t'])
        self.BATCH_SIZE = int(tp['batch_size'])
        self.Label_number = int(tp['label_number'])
        self.features_size = int(tp['features_size'])
        self.weights = list(map(float, tp['label_weights'].split(',')))

    #   change learning rate
    def get_opt(self, acc, bestacc, count, lr_rate):
        if acc > bestacc:
            count = 0
        else:
            count += 1
        print('hold count : ' + str(count))
        if count == self.hold_threshold:
            count = 0
            lr_rate = lr_rate * self.lr_r
            with open('./Log/model_2.txt', 'a+') as r:
                line = '---------- have update learning rate : ' + str(lr_rate) + ' -----------\n'
                r.write(line)
        return count, lr_rate

    #   save model parameters and other information
    def save_model(self, model, Y_T, Y_P, cp, p_report, t, tacc, bestacc, test_loss, bestloss):
        #   get predicted label distribution
        rate = [0.0] * 11
        y = np.array(Y_T)
        cp = np.array(cp)
        for i in range(len(rate)):
            y_i = np.argwhere(y == i)
            rate[i] = (cp[y_i].tolist().count([1]) + cp[y_i].tolist().count([-1])) / len(cp)
        # when get best acc,save the model and Record results
        if tacc > bestacc:
            torch.save(model.state_dict(), './Log/model_2_bestacc.pkl')
            bestacc = tacc
            with open('./Log/model_2_bestacc.txt', 'w') as rw:
                rw.write(p_report)
                rw.write('\nTest acc: ' + str(tacc))
                rw.write('\n+-1 rate : ' + str(t))
                rw.write('\n+-1 distribution : ' + str(rate))
                rw.write('\n' + ','.join(map(str, Y_T)))
                rw.write('\n' + ','.join(map(str, Y_P)))
        # when get best loss,save the model
        if test_loss < bestloss:
            torch.save(model.state_dict(), './Log/model_2_bestloss.pkl')
            bestloss = test_loss
            with open('./Log/model_2_bestloss.txt', 'w') as rw:
                rw.write('\nBest loss : ' + str(bestloss))
        # save the last model
        torch.save(model.state_dict(), './Log/model_2_last.pkl')
        with open('./Log/model_2_last.txt', 'w') as lw:
            lw.write(p_report)
            lw.write('\nTest acc: ' + str(tacc))
            lw.write('\n+-1 rate : ' + str(t))
            lw.write('\n+-1 distribution : ' + str(rate))
            lw.write('\n' + ','.join(map(str, Y_T)))
            lw.write('\n' + ','.join(map(str, Y_P)))
        # Record the results of each validation
        with open('./Log/model_2.txt', 'a+') as r:
            line = 'Test acc: ' + str(tacc) + '\t' + 'Loss: ' + str(test_loss) + '+-1 rate : ' + str(t) + '\n'
            r.write(line)

    def Train(self):
        print('This is Train')
        count = 0
        lr_rate = self.lr_rate
        bestacc = 0.0
        bestloss = 100.0
        #   Run Training, Resnet+Attention and Bilstm+CRF is all available,just replace the corresponding model.
        model = ResNet18(batch_size=self.BATCH_SIZE, weight=self.weights, feature_size=self.features_size)
        # model = BiLstm_CRF(hidden_dim=1024,features_number=features_size,layer_num=2,batch_size=BATCH_SIZE,label_number=Label_number)

        #   pretrain and Transfer Learning
        if self.pretrained == 'True':
            if os.path.exists('./Log/pre_model_2_bestacc_4label.pkl'):
                pretrained_dict = torch.load('./Log/pre_model_2_bestacc_4label.pkl')
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict((model_dict))
                for k, v in model.named_parameters():
                    # if k in pretrained_dict:
                    if 'layer' in k:
                        v.requires_grad = False
                for k, v in model.named_parameters():
                    print(k)
                    print(v.requires_grad)
                print('Loading pretrained parameters completed ! ')
            else:
                print('pretrained parameters file not exist ! ')
        #   If program is terminated and there is model file of last validation,then model will load it and continue train.
        if os.path.exists('./Log/model_2_last.pkl'):
            model.load_state_dict(torch.load('./Log/model_2_bestacc.pkl'))
            with open('./Log/model_2.txt', 'a+') as r:
                line = '----------- continue train, learning rate : ' + str(lr_rate) + ' -----------\n'
                r.write(line)
            print('continue train ! ')

        if torch.cuda.is_available():
            model.cuda()
        train_file = 'nce' + self.NCE + '_train.txt'
        validation_file = 'nce' + self.NCE + '_validation.txt'
        Data = data('./Data', self.Label_number, discretization=0, train_file=train_file,
                    validation_file=validation_file)
        if not os.path.exists('./Log'):
            os.mkdir('./Log')

        for epoch in range(self.epoch):
            #   get data
            Train_data, Train_label, Train_length, Test_data, Test_label, Test_length = Data.GetData(self.BATCH_SIZE)
            ep_count = 0
            ep_validation = int(len(Train_data) / 3)
            #   train
            for b_index, b_data in enumerate(Train_data):
                input_features = torch.tensor(b_data).cuda()
                ions_label = torch.tensor(Train_label[b_index]).cuda()
                batch_length = torch.tensor(Train_length[b_index]).cuda()
                model.zero_grad()
                y_true, y_pred, results, loss, _ = model(input_features.permute(0, 2, 1), ions_label, batch_length)
                loss.backward()
                optimizer = optim.Adam(model.parameters(), lr=lr_rate, weight_decay=1e-4)
                optimizer.step()
                ep_count += 1
                if ep_count % ep_validation == 0:
                    with torch.no_grad():
                        Y_T = []
                        Y_P = []
                        test_loss = 0.0
                        for T_index, T_data in enumerate(Test_data):
                            t_input_features = torch.tensor(T_data).cuda()
                            t_ions_level = torch.tensor(Test_label[T_index]).cuda()
                            t_batch_length = torch.tensor(Test_length[T_index]).cuda()
                            if t_input_features.shape[0] != self.BATCH_SIZE:
                                continue
                            y_true, y_pred, results, loss, _ = model(t_input_features.permute(0, 2, 1), t_ions_level,
                                                                     t_batch_length)
                            test_loss += loss
                            Y_T.extend(y_true)
                            Y_P.extend(y_pred)
                        test_loss = test_loss / len(Test_data)
                        tags_name = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']
                        p_report = classification_report(Y_T, Y_P, target_names=tags_name)
                        cp = (np.array(Y_T) - np.array(Y_P)).tolist()
                        tacc = cp.count(0) / len(cp)
                        print('Test acc: ' + str(tacc))
                        count, lr_rate = self.get_opt(tacc, bestacc, count, lr_rate)
                        t = (cp.count(1) + cp.count(-1)) / len(cp)
                        print('+-1 rate : ' + str(t))
                        self.save_model(model=model, Y_T=Y_T, Y_P=Y_P, cp=cp, p_report=p_report, t=t, tacc=tacc,
                                        bestacc=bestacc, test_loss=test_loss, bestloss=bestloss)
            with open('./Log/model_2.txt', 'a+') as r:
                r.write('%%%%%%% Now epoch : ' + str(epoch) + ' %%%%%%%\n')
            del Train_data, Train_label, Train_length, Test_data, Test_label, Test_length
            gc.collect()

    def Test(self):
        #   Run Testing
        print('This is Test')
        model = ResNet18(self.BATCH_SIZE, weight=self.weights, feature_size=self.features_size)
        model.load_state_dict(torch.load('./Log/model_2_bestacc.pkl'))
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        test_file = 'nce' + self.NCE + '_test.txt'
        test_data = data('./Data', self.Label_number, run_model='Test', test_file=test_file)
        Test_data, Test_label, Test_length, _, _, _ = test_data.GetData(self.BATCH_SIZE)
        print('Test data number: ' + str(len(Test_length) * self.BATCH_SIZE))
        with torch.no_grad():
            Y_T = []
            Y_P = []
            Results = []
            Attention = []
            for T_index, T_data in tqdm(enumerate(Test_data)):
                t_input_features = torch.tensor(T_data).cuda()
                t_ions_level = torch.tensor(Test_label[T_index]).cuda()
                t_batch_length = torch.tensor(Test_length[T_index]).cuda()
                y_true, y_pred, results, loss, _attn = model(t_input_features.permute(0, 2, 1), t_ions_level,
                                                             t_batch_length)
                Y_T.extend(y_true)
                Y_P.extend(y_pred)
                Results.extend(results)
                Attention.extend(_attn)
            tags_name = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']
            p_report = classification_report(Y_T, Y_P, target_names=tags_name)
            cp = (np.array(Y_T) - np.array(Y_P)).tolist()
            tacc = cp.count(0) / len(cp)
            print('Test acc: ' + str(tacc))
            t = (cp.count(1) + cp.count(-1)) / len(cp)
            print(t)
            rate = [0.0] * 11
            y = np.array(Y_T)
            cp = np.array(cp)
            for i in range(len(rate)):
                y_i = np.argwhere(y == i)
                rate[i] = (cp[y_i].tolist().count([1]) + cp[y_i].tolist().count([-1])) / len(cp)
            print(rate)
            #   save Test results
            with open('./Log/model_2_test.txt', 'w') as lw:
                lw.write(p_report)
                lw.write('\nTest acc: ' + str(tacc))
                lw.write('\n+-1 rate : ' + str(t))
                lw.write('\n+-1 distribution : ' + str(rate))
                lw.write('\n' + str(Results))

    def Run(self):
        if self.run_model == 'Train':
            self.Train()
        else:
            self.Test()


if __name__ == '__main__':
    model = Pscore_Model()
    model.Run()
