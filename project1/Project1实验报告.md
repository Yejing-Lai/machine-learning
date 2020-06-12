## 项目一实验报告

数据集：Car Evaluation Data Set

数据集网址：http://archive.ics.uci.edu/ml/datasets/Car+Evaluation

训练方法：神经网络

目的：根据车辆的情况来预测车辆的状态

#### 数据处理分析

##### 四个类别：

- unacc（车辆状况很差）

- acc（车辆状况一般）
- good（车辆状况好）
- vgood（车辆状况很好）

##### 六个属性：

- buying（购买价格）: vhigh, high, med, low.
- maint（维护价格）: vhigh, high, med, low.
- doors（门的数量）: 2, 3, 4, 5more.
- persons（核载人数）: 2, 4, more. 
- lug_boot（空间大小）small, med, big.
- safety（安全性）: low, med, high.

##### 展示前五条数据：

![image-20200505000349305](C:\Users\Era\AppData\Roaming\Typora\typora-user-images\image-20200505000349305.png)

总共有1728条数据。

##### 数据预处理

由于神经网络只能够读取数字数据，而原数据中有字符串数据，在这次实验中，把每个数据转转为one-hot形式。也可以用不同的数字代替字符串数据，如buying这个属性里，有四种情况（vhigh, high, med, low），可以用以下的代替方式（vhigh=1, high=2, med=3, low=4）。但若使用不同数字进行转换，则会产生距离上的差距。但实际中这些情况不一定会有距离的差别。所以在本次实验中，假设每种都是独立的，不能相互比较的，所以采用了one-hot形式把原来的字符串数据转为数字数据。

### 方法分析

本次实验中使用的是神经网络方法。添加了一个含有128个神经元的隐藏层，经过100个epoch，可获得约等于98.5%的准确率。另外，尝试添加两个含有128个神经元的隐藏层，可获得将近99.9%的准确率（取多次实验结果的平均）。,其中输入x的维数为4+4+4+3+3+3=21，输出out的维数为4。

### 发现与结论

一般来说，神经网络越深，准确率越高，拟合程度越高，但容易造成过拟合。在本次实验中，数据集较小，使用了较小的网络对数据进行拟合，可以获得很高的准确率。其中，尝试了0.1;0.01;0.001;的学习率，也可以获得很高的准确率。对于这类较小的数据集来说，小的epoch和小的神经网络都可以更快更准确的获得最终的结果。

### 源码

```python
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F

#读取数据
def load_data():
    col_names = ['buying','maint','doors','persons','lug_boot','safety','class']
    data = pd.read_csv("car.csv",names = col_names)
    return data
#把数据转为one-hot向量
def convert2onehot(data):
    return pd.get_dummies(data, columns=data.columns)
    
data = load_data()
new_data = convert2onehot(data)
new_data = new_data.values.astype(np.float32)

#把数据打乱后，将训练数据和测试数据按7:3的比例分开
np.random.shuffle(new_data)
seq = int(0.7*len(new_data))
train_data = new_data[:seq]
test_data = new_data[seq:]

import torch
#获取训练数据
train_X = torch.from_numpy(train_data[:,:21])
train_Y = train_data[:, 21:]
train_Y = np.argmax(train_Y,axis=1)
train_Y = torch.from_numpy(train_Y)
#获取测试数据
test_X = torch.from_numpy(test_data[:, :21])
test_Y = test_data[:, 21:]
test_Y = np.argmax(test_Y,axis=1)
test_Y = torch.from_numpy(test_Y)

#定义神经网络
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

#使用交叉熵损失函数
loss_fuc = nn.CrossEntropyLoss()
#优化器使用Adam
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)
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
```

### 输出结果

epoch:0  loss:1.5562983751296997

epoch:1  loss:1.9791654348373413

epoch:2  loss:1.2256656885147095

epoch:3  loss:0.6051774621009827

epoch:4  loss:0.5699688792228699

epoch:5  loss:0.5740585923194885

epoch:6  loss:0.5503944158554077

epoch:7  loss:0.5283333659172058

epoch:8  loss:0.5062553286552429

epoch:9  loss:0.48532435297966003

epoch:10  loss:0.46827059984207153

epoch:11  loss:0.4512771964073181

epoch:12  loss:0.43160679936408997

epoch:13  loss:0.4113371670246124

epoch:14  loss:0.3916374444961548

epoch:15  loss:0.37213456630706787

epoch:16  loss:0.35351431369781494

epoch:17  loss:0.3369133174419403

epoch:18  loss:0.32308241724967957

epoch:19  loss:0.31083881855010986

epoch:20  loss:0.29766952991485596

epoch:21  loss:0.2848348915576935

epoch:22  loss:0.2741234004497528

epoch:23  loss:0.26586076617240906

epoch:24  loss:0.2584945857524872

epoch:25  loss:0.25072452425956726

epoch:26  loss:0.2423902153968811

epoch:27  loss:0.23395167291164398

epoch:28  loss:0.225777268409729

epoch:29  loss:0.21826386451721191

epoch:30  loss:0.21159471571445465

epoch:31  loss:0.20520952343940735

epoch:32  loss:0.19805032014846802

epoch:33  loss:0.18998338282108307

epoch:34  loss:0.1816178560256958

epoch:35  loss:0.17359596490859985

epoch:36  loss:0.16553595662117004

epoch:37  loss:0.15679113566875458

epoch:38  loss:0.1471378356218338

epoch:39  loss:0.1370457261800766

epoch:40  loss:0.1265970766544342

epoch:41  loss:0.11623997986316681

epoch:42  loss:0.10618164390325546

epoch:43  loss:0.09683703631162643

epoch:44  loss:0.08840610086917877

epoch:45  loss:0.08029074221849442

epoch:46  loss:0.07266366481781006

epoch:47  loss:0.06653786450624466

epoch:48  loss:0.06163778901100159

epoch:49  loss:0.05696171522140503

epoch:50  loss:0.052619513124227524

epoch:51  loss:0.048986293375492096

epoch:52  loss:0.04563149809837341

epoch:53  loss:0.04231343790888786

epoch:54  loss:0.039551664143800735

epoch:55  loss:0.03720324486494064

epoch:56  loss:0.0349460206925869

epoch:57  loss:0.032961055636405945

epoch:58  loss:0.03129559010267258

epoch:59  loss:0.02969338931143284

epoch:60  loss:0.028127312660217285

epoch:61  loss:0.026827914640307426

epoch:62  loss:0.025661420077085495

epoch:63  loss:0.024495115503668785

epoch:64  loss:0.023375794291496277

epoch:65  loss:0.02230047807097435

epoch:66  loss:0.021231558173894882

epoch:67  loss:0.020194627344608307

epoch:68  loss:0.019268078729510307

epoch:69  loss:0.018434612080454826

epoch:70  loss:0.01761970855295658

epoch:71  loss:0.016854265704751015

epoch:72  loss:0.01616695150732994

epoch:73  loss:0.015499096363782883

epoch:74  loss:0.014826271682977676

epoch:75  loss:0.014184603467583656

epoch:76  loss:0.013584816828370094

epoch:77  loss:0.013011538423597813

epoch:78  loss:0.012471738271415234

epoch:79  loss:0.011983084492385387

epoch:80  loss:0.011528928764164448

epoch:81  loss:0.011088628321886063

epoch:82  loss:0.010669190436601639

epoch:83  loss:0.010277328081429005

epoch:84  loss:0.00991386454552412

epoch:85  loss:0.009566428139805794

epoch:86  loss:0.009236890822649002

epoch:87  loss:0.008933615870773792

epoch:88  loss:0.008647453971207142

epoch:89  loss:0.008369436487555504

epoch:90  loss:0.008103197440505028

epoch:91  loss:0.007856356911361217

epoch:92  loss:0.00762347923591733

epoch:93  loss:0.007401264272630215

epoch:94  loss:0.007191121578216553

epoch:95  loss:0.006990205496549606

epoch:96  loss:0.006800278555601835

epoch:97  loss:0.006617800798267126

epoch:98  loss:0.00644304882735014

epoch:99  loss:0.0062776245176792145

acc: 0.9903660886319846