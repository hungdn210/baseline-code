#!/usr/bin/env python3

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable

from data import data_preprocess, data_trans
from modelbase import STA_LSTM as Net
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib
# from modelbase import SA_LSTM as Net
# from modelbase import TA_LSTM as Net
# from modelbase import LSTM as Net
# from modelbase import FCN as Net
# from modelbase import SVM as Net

'''****************************initialization*******************************''' 
IN_DIM = 32 * 26       # 因变量 TX144，CH96，HH120
SEQUENCE_LENGTH = 32   # 时间序列长度，即为回溯期

LSTM_IN_DIM = int(IN_DIM/SEQUENCE_LENGTH)     # LSTM的input大小,等于总的变量长度/时间序列长度
LSTM_HIDDEN_DIM = 300  # LSTM隐状态的大小

OUT_DIM = 7 * 26            # 输出大小

LEARNING_RATE = 0.05 # learning rate
WEIGHT_DECAY = 1e-6    # L2惩罚项

BATCH_SIZE = 200        # batch size

EPOCHES = 100    # epoch大小

TRAIN_PER = 0.80 # 训练集占比
VALI_PER = 0.0 # 验证集占比

# 判断是否采用GPU加速
# USE_GPU = torch.cuda.is_available()
USE_GPU = False

'''****************************data prepration*******************************''' 
# 准备好训练和测试数据
dp = data_preprocess(file_path = '../data/dataset/Water_Discharge_STA_Normalized.csv', train_per = TRAIN_PER, vali_per = VALI_PER, in_dim = IN_DIM)

raw_data = dp.load_data()
# print('数据导入完成')

(train_data,train_groundtruth),(vali_data,vali_groundtruth),(test_data,test_groundtruth) = dp.split_data(raw_data = raw_data, _type = 'linear')
# print('数据分割完成')

# 设置对数据进行的转换方式，transform.compose的作用是将多个transform组合到一起进行使用
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=(0,0,0),std=(1,1,1))])
# print('数据转换为tensor')

# data_trans返回的值是一个字典，内部包含数据和真值{'inputs':inputs,'groundtruth':groundtruths}

# 准备训练集
train_data_trans = data_trans(train_data,train_groundtruth,transform)

train_dataloader = torch.utils.data.DataLoader(train_data_trans,
                                           batch_size =BATCH_SIZE,
                                           shuffle = True,
                                           num_workers = 4)
# print('训练集准备完毕')

# 准备测试集
test_data_trans = data_trans(test_data, test_groundtruth,transform)

test_dataloader = torch.utils.data.DataLoader(test_data_trans,
                                           batch_size = BATCH_SIZE,
                                           shuffle = False,
                                           num_workers = 4)
# print('测试集准备完毕')


'''****************************model prepration*******************************''' 
# 将网络参数导入网络
net = Net(IN_DIM,SEQUENCE_LENGTH,LSTM_IN_DIM,LSTM_HIDDEN_DIM,OUT_DIM,USE_GPU)
# print('网络模型准备完毕')

# 判断GPU是否可用，如果可用则将net变成可用GPU加速的net
if USE_GPU:
    net = net.cuda()
    # print('本次实验使用GPU加速')
else:
    pass
    # print('本次实验不使用GPU加速')

# 使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9
# optimizer = optim.SGD(net.parameters(), lr= LEARNING_RATE, momentum=0.9) 
# 根据梯度调整参数数值，Adam算法
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 学习率根据训练的次数进行调整
adjust_lr = optim.lr_scheduler.MultiStepLR(optimizer,
                                     milestones=[i*10 for i in range(EPOCHES//10)],
                                     gamma=0.5)

# 定义训练损失函数&测试误差函数
# loss_criterion = nn.SmoothL1Loss()
loss_criterion = nn.MSELoss()
error_criterion = nn.MSELoss()


def train(verbose = False):

    net.train()
    loss_list = []

    for i,data in enumerate(train_dataloader):
       
        inputs = data['inputs']
        groundtruths = data['groundtruths']     
        
        if USE_GPU:
            inputs = Variable(inputs).cuda()
            groundtruths = Variable(groundtruths).cuda()
            
        else:
            inputs = Variable(inputs)
            groundtruths = Variable(groundtruths)
        
        #将参数的grad值初始化为0
        optimizer.zero_grad()

        #获得网络输出结果
        out = net(inputs)

        #根据真值计算损失函数的值
        loss = loss_criterion(out,groundtruths)

        #通过优化器优化网络
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
      
    return loss_list


def test():

    error = 0.0
    predictions = []
    test_groundtruths = []

    net.eval()

    for i, data in enumerate(test_dataloader):
        inputs = data['inputs']
        groundtruths = data['groundtruths']     

        if USE_GPU:
            inputs = Variable(inputs).cuda()
            groundtruths = Variable(groundtruths).cuda()
        else:
            inputs = Variable(inputs)
            groundtruths = Variable(groundtruths)

        out = net(inputs)
        error += (error_criterion(out, groundtruths).item() * groundtruths.size(0))

        if USE_GPU:
            predictions.extend(out.cpu().data.numpy().tolist())
            test_groundtruths.extend(groundtruths.cpu().data.numpy().tolist())
        else:
            predictions.extend(out.data.numpy().tolist())
            test_groundtruths.extend(groundtruths.data.numpy().tolist())

    # Load saved scaler for denormalization
    scaler = joblib.load('../data/output/output_scaler.pkl')

    predictions = scaler.inverse_transform(np.array(predictions))
    test_groundtruths = scaler.inverse_transform(np.array(test_groundtruths))

    mse = mean_squared_error(test_groundtruths, predictions)
    rmse = mean_squared_error(test_groundtruths, predictions, squared=False)
    mae = mean_absolute_error(test_groundtruths, predictions)
    r2 = r2_score(test_groundtruths, predictions)

    print(f"\nEvaluation Metrics:")
    print(f"  MSE  = {mse:.6f}")
    print(f"  RMSE = {rmse:.6f}")
    print(f"  MAE  = {mae:.6f}")
    print(f"  R2   = {r2:.6f}")

    return predictions, test_groundtruths



def main():
    train_start = time.time()
    loss_recorder = []

    print('starting training... ')
    for epoch in range(EPOCHES):
        adjust_lr.step()
        loss_list = train(verbose=True)
        loss_recorder.append(np.mean(loss_list))
        print('epoch = %d,loss = %.5f' % (epoch+1, np.mean(loss_list)))

    print('training time = {}s'.format(int((time.time() - train_start))))

    test_start = time.time()
    predictions, test_groundtruth = test()

    print('test time = {}s'.format(int((time.time() - test_start) + 1.0)))

    result = pd.DataFrame(data={'Prediction': predictions.flatten(), 'GroundTruth': test_groundtruth.flatten()})
    os.makedirs('../data/output', exist_ok=True)
    result.to_csv('../data/output/out_t+1.csv', index=False)

    torch.save(net, '../models/sta_lstm_t+1.pth')


if __name__ == '__main__':
    main()
