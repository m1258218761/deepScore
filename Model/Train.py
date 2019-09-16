import torch
import torch.nn as nn
import torch.optim as optim
from data_util import data
import os
import gc
from sklearn.metrics import classification_report
import numpy as np
from  Resnet_model import ResNet18
from tqdm import tqdm

NCE = '30'
run_model = 'Train'         ## run model:Train or Test
pretrained = False         ##pretrained:True or False

##Model parameters
BATCH_SIZE = 16             ##batch size
Label_number = 4            ##label 2:predict b1+,y1+;   label 4:predict b1+,b2+,y1+,y2+
features_size = 105         ##feature size of every Peptide bond

#   label weight for cross entropy
weights2 = [0.4008, 0.2264, 0.1307, 0.0719, 0.0446, 0.0303, 0.0219, 0.0163, 0.0123, 0.0092, 0.0349]
weights4 = [0.6550, 0.1455, 0.0718, 0.0385, 0.0236, 0.0158, 0.0113, 0.0084, 0.0063, 0.0047, 0.0190]



def Train():
    #   adjust learning rate
    global count,lr_rate
    count = 0
    lr_rate = 1e-4
    def get_opt(acc,bestacc):
        global count
        global lr_rate
        if acc > bestacc:
            count = 0
        else:
            count += 1
        print('count : ' + str(count))
        if count == 30:
            count = 0
            lr_rate = lr_rate*0.1
            with open('./Log/model_2.txt', 'a+') as r:
                line = '---------- have update learning rate : ' + str(lr_rate) + ' -----------\n'
                r.write(line)
    #   Run Training
    model = ResNet18(batch_size=BATCH_SIZE,weight=weights4,feature_size=features_size)
    if pretrained:
        #   pretrain and Transfer Learning
        if os.path.exists('./Log/pre_model_2_bestacc_4label.pkl'):
            pretrained_dict = torch.load('./Log/pre_model_2_bestacc_4label.pkl')
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict((model_dict))
            for k,v in model.named_parameters():
                # if k in pretrained_dict:
                if 'layer' in k:
                    v.requires_grad = False
            for k,v in model.named_parameters():
                print(k)
                print(v.requires_grad)
            print('Loading pretrained parameters completed ! ')
        else:
            print('pretrained parameters file not exist ! ')

    if os.path.exists('./Log/model_2_bestacc.pkl'):
        model.load_state_dict(torch.load('./Log/model_2_bestacc.pkl'))
        with open('./Log/model_2.txt', 'a+') as r:
            line = '----------- continue train, learning rate : ' + str(lr_rate) + ' -----------\n'
            r.write(line)
        print('continue train ! ')

    if torch.cuda.is_available():
        model.cuda()
    train_file = 'nce'+NCE+'_train.txt'
    validation_file = 'nce'+NCE+'_validation.txt'
    Data = data('./Data',Label_number,discretization=0,train_file=train_file,validation_file=validation_file)
    if os.path.exists('./Log'):
        pass
    else:
        os.mkdir('./Log')

    bestacc = 0.0
    bestloss = 50.0
    for epoch in range(200):
        #   get data
        Train_data, Train_label, Train_length, Test_data, Test_label, Test_length = Data.GetData(BATCH_SIZE)
        ep_count = 0
        ep_validation = int(len(Train_data)/3)
        #   train
        for b_index,b_data in enumerate(Train_data):
            input_features = torch.tensor(b_data).cuda()
            ions_label = torch.tensor(Train_label[b_index]).cuda()
            batch_length = torch.tensor(Train_length[b_index]).cuda()
            model.zero_grad()
            y_true,y_pred,results,loss,_ = model(input_features.permute(0,2,1),ions_label,batch_length)
            loss.backward()
            optimizer = optim.Adam(model.parameters(),lr=lr_rate,weight_decay=1e-4)
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
                        y_true, y_pred, results, loss, _ = model(t_input_features.permute(0,2,1) , t_ions_level,t_batch_length)
                        test_loss += loss
                        Y_T.extend(y_true)
                        Y_P.extend(y_pred)
                    test_loss = test_loss/len(Test_data)
                    tags_name = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9','c10']
                    p_report = classification_report(Y_T,Y_P,target_names=tags_name)
                    cp = (np.array(Y_T) - np.array(Y_P)).tolist()
                    tacc = cp.count(0)/len(cp)
                    print('Test acc: ' + str(tacc))
                    get_opt(tacc,bestacc)
                    t = (cp.count(1) + cp.count(-1)) / len(cp)
                    print('+-1 rate : ' + str(t))

                    #   get predicted label distribution
                    rate = [0.0] * 11
                    y = np.array(Y_T)
                    cp = np.array(cp)
                    for i in range(len(rate)):
                        y_i = np.argwhere(y == i)
                        rate[i] = (cp[y_i].tolist().count([1]) + cp[y_i].tolist().count([-1]))/len(cp)
                    #   when get best acc,save the model and Record results
                    if tacc > bestacc:
                        torch.save(model.state_dict(),'./Log/model_2_bestacc.pkl')
                        bestacc = tacc
                        with open('./Log/model_2_bestacc.txt','w') as rw:
                            rw.write(p_report)
                            rw.write('\nTest acc: ' + str(tacc))
                            rw.write('\n+-1 rate : ' + str(t))
                            rw.write('\n+-1 distribution : ' + str(rate))
                            rw.write('\n' + ','.join(map(str,Y_T)))
                            rw.write('\n' + ','.join(map(str,Y_P)))
                    #   when get best loss,save the model
                    if test_loss < bestloss:
                        torch.save(model.state_dict(),'./Log/model_2_bestloss.pkl')
                        bestloss = test_loss
                        with open('./Log/model_2_bestloss.txt','w') as rw:
                            rw.write('\nBest loss : ' + str(bestloss))
                    #   save the last model
                    torch.save(model.state_dict(),'./Log/model_2_last.pkl')
                    with open('./Log/model_2_last.txt','w') as lw:
                        lw.write(p_report)
                        lw.write('\nTest acc: ' + str(tacc))
                        lw.write('\n+-1 rate : ' + str(t))
                        lw.write('\n+-1 distribution : ' + str(rate))
                        lw.write('\n' + ','.join(map(str, Y_T)))
                        lw.write('\n' + ','.join(map(str, Y_P)))
                    #   Record the results of each validation
                    with open('./Log/model_2.txt','a+') as r:
                        line = 'Test acc: ' + str(tacc) + '\t'+'Loss: '+ str(test_loss)+'+-1 rate : '+str(t) + '\n'
                        r.write(line)
        with open('./Log/model_2.txt','a+') as r:
            r.write('%%%%%%% Now epoch : ' + str(epoch) +' %%%%%%%\n')
        del Train_data, Train_label, Train_length, Test_data, Test_label, Test_length
        gc.collect()
def Test():
    #   Run Testing
    print('This is Test')
    model = ResNet18(BATCH_SIZE,weight=weights4,feature_size=features_size)
    model.load_state_dict(torch.load('./Log/model_2_bestacc.pkl'))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    test_file = 'nce'+NCE+'_test.txt'
    test_data = data('./Data', Label_number,run_model='Test',test_file=test_file)
    Test_data, Test_label, Test_length,_,_,_ = test_data.GetData(BATCH_SIZE)
    print('Test data number: ' + str(len(Test_length) * BATCH_SIZE))
    with torch.no_grad():
        Y_T = []
        Y_P = []
        Results = []
        Attention = []
        for T_index, T_data in tqdm(enumerate(Test_data)):
            t_input_features = torch.tensor(T_data).cuda()
            t_ions_level = torch.tensor(Test_label[T_index]).cuda()
            t_batch_length = torch.tensor(Test_length[T_index]).cuda()
            y_true,y_pred,results,loss,_attn = model(t_input_features.permute(0,2,1) ,t_ions_level,t_batch_length)
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

if __name__ == '__main__':
    if run_model=='Train':
        Train()
    else:
        Test()