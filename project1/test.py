import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F

def load_data():
    col_names = ['buying','maint','doors','persons','lug_boot','safety','class']
    data = pd.read_csv("car.csv",names = col_names)
    return data

def convert2onehot(data):
    return pd.get_dummies(data, columns=data.columns)
    
data = load_data()
new_data = convert2onehot(data)
new_data = new_data.values.astype(np.float32)

np.random.shuffle(new_data)
seq = int(0.7*len(new_data))
train_data = new_data[:seq]
test_data = new_data[seq:]

import torch
train_X = torch.from_numpy(train_data[:,:21])
#train_Y = torch.from_numpy(train_data[:, 21:])
train_Y = train_data[:, 21:]
train_Y = np.argmax(train_Y,axis=1)
train_Y = torch.from_numpy(train_Y)
#train_Y = np.argmax(train_Y)
test_X = torch.from_numpy(test_data[:, :21])
test_Y = test_data[:, 21:]
test_Y = np.argmax(test_Y,axis=1)
test_Y = torch.from_numpy(test_Y)

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.linear1 = nn.Linear(21,128)
        #self.linear2 = nn.Linear(128,128)
        self.linear3 = nn.Linear(128,4)
        
    def forward(self,x):
        x = F.relu(self.linear1(x))
        #x = F.relu(self.linear2(x))
        out = self.linear3(x)
        return out

model = net()

loss_fuc = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.1)
for epoch in range(100):
    ouput = model(train_X)
    loss = loss_fuc(ouput,train_Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("epoch:{}    loss:{}".format(epoch,loss))

test_ouput = model(test_X)
pred_y = torch.max(test_ouput,1)[1].data.numpy()
acc = float((pred_y == test_Y.data.numpy()).astype(int).sum()) / float(test_Y.size(0))
print('acc:',acc)