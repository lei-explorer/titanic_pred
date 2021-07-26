# -*- coding:utf-8 -*-
import torch
import numpy as np
from math import isnan
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from torch.utils.data import Dataset,DataLoader
from sklearn import linear_model
torch.set_default_tensor_type(torch.DoubleTensor)
mpl.rcParams["font.family"] = "SimHei"  # 添加中文字体名称
mpl.rcParams["axes.unicode_minus"]=False # 由于更改了字体导致显示不出负号，此设置用来正常显示负号
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
pd.set_option('display.max_columns', None) #显示所有列
#----------------4步-------------------
#1、准备数据集

class train_Dataset(Dataset):
    def __init__(self,filepath):
        self.data_train = pd.read_csv(filepath,sep=',',header=0,index_col=None)
        self.len = len(self.data_train)
        self.data_train, rfr = self.set_missing_ages(self.data_train)
        self.data_train = self.set_Cabin_type(self.data_train)

        # 对类目型的特征因子化
        dummies_Cabin = pd.get_dummies(self.data_train['Cabin'], prefix='Cabin')
        dummies_Embarked = pd.get_dummies(self.data_train['Embarked'], prefix='Embarked')
        dummies_Sex = pd.get_dummies(self.data_train['Sex'], prefix='Sex')
        dummies_Pclass = pd.get_dummies(self.data_train['Pclass'], prefix='Pclass')
        df = pd.concat([self.data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
        df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

        # 将Age和Fare两个属性一些变化幅度较大的特征化到[-1, 1]之内
        scaler = preprocessing.StandardScaler()
        age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))
        df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1), age_scale_param)
        fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
        df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)

        train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*',axis=1)
        train_np = train_df.values

        # y即Survival结果
        self.y_data = train_np[:, 0]
        # X即特征属性值
        self.x_data = train_np[:, 1:]
        # print(self.x_data.shape)
        # print(self.y_data)

        # fit到RandomForestRegressor之中
        # clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        # clf.fit(self.x_data,self.y_data)


    ### 使用 RandomForestClassifier 填补缺失的年龄属性
    def set_missing_ages(self,df):
        # 把已有的数值型特征取出来丢进Random Forest Regressor中
        age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

        # 乘客分成已知年龄和未知年龄两部分
        known_age = age_df[age_df.Age.notnull()].values
        unknown_age = age_df[age_df.Age.isnull()].values

        # y即目标年龄
        y = known_age[:, 0]
        # X即特征属性值
        X = known_age[:, 1:]
        # fit到RandomForestRegressor之中
        rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
        rfr.fit(X, y)
        # 用得到的模型进行未知年龄结果预测
        predictedAges = rfr.predict(unknown_age[:, 1:])
        # 用得到的预测结果填补原缺失数据
        df.loc[(df.Age.isnull()), 'Age'] = predictedAges
        return df, rfr

    def set_Cabin_type(self,df):
        df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
        df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
        return df

    def __getitem__(self, index):
        return self.x_data[index] , self.y_data[index]

    def __len__(self):
        return self.len

class test_dataset(Dataset):
    def __init__(self,filepath):
        self.data_test = pd.read_csv(filepath, sep=',', header=0, index_col=None)
        self.len = len(self.data_test)
        # print(self.len)
        self.data_test, rfr = self.set_missing_ages(self.data_test)
        self.data_test = self.set_Cabin_type(self.data_test)
        # print(self.data_test)


        # 对类目型的特征因子化
        dummies_Cabin = pd.get_dummies(self.data_test['Cabin'], prefix='Cabin')
        dummies_Embarked = pd.get_dummies(self.data_test['Embarked'], prefix='Embarked')
        dummies_Sex = pd.get_dummies(self.data_test['Sex'], prefix='Sex')
        dummies_Pclass = pd.get_dummies(self.data_test['Pclass'], prefix='Pclass')
        df = pd.concat([self.data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
        df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

        # 将Age和Fare两个属性一些变化幅度较大的特征化到[-1, 1]之内
        scaler = preprocessing.StandardScaler()
        age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
        df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
        fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
        df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)

        train_df = df.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*', axis=1)

        train_np = train_df.values
        # print(train_np.shape)
        # X即特征属性值
        self.x_data = train_np

    def set_missing_ages(self,df):
        # 把已有的数值型特征取出来丢进Random Forest Regressor中
        age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        # 乘客分成已知年龄和未知年龄两部分
        known_age = age_df[age_df.Age.notnull()].values
        unknown_age = age_df[age_df.Age.isnull()].values
        # print(known_age.shape)
        # y即目标年龄
        y = known_age[:, 0]
        # X即特征属性值
        X = known_age[:, 1:]
        for i in range(len(X[:,0])):
            if isnan(X[i,0]):
                X[i,0] = X[i-1,0]

            # else:
            #     print(type(X[i,0]))
        # fit到RandomForestRegressor之中
        rfr = RandomForestRegressor(random_state=0, n_estimators=1000, n_jobs=-1)
        rfr.fit(X, y)
        # 用得到的模型进行未知年龄结果预测
        predictedAges = rfr.predict(unknown_age[:, 1:])
        # 用得到的预测结果填补原缺失数据
        df.loc[(df.Age.isnull()), 'Age'] = predictedAges
        return df, rfr

    def set_Cabin_type(self,df):
        df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
        df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
        return df

    def test_sample(self,index):
        return self.x_data[index],self.data_test.loc[index,'PassengerId']

    def length(self):
        return len(self.x_data)

#2、构建模型
class LinearModel(torch.nn.Module):  #继承torch.nn.Module类
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear1 = torch.nn.Linear(14,6)
        self.linear2 = torch.nn.Linear(6,1)
        self.sigmod = torch.nn.Sigmoid() # Sigmoid()是Module下的一个模块
        self.relu =torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self,x):
        # print('original_x=',x)
        x = self.sigmod(self.linear1(x))
        x = self.sigmod(self.linear2(x))
        # x = self.softmax(x)
        x = x.squeeze(-1)
        return x

model = LinearModel()

#3、构造损失函数和优化器
criterion = torch.nn.BCELoss(reduction='sum')  #size_average是否对loss求平均，即除以N,BCELoss是交叉熵求损失
optimizer = torch.optim.RMSprop(model.parameters(),lr=0.1) #model.parameters()是取出model里所有要优化（更新）的参数,lr学习率

# 4、写训练循环
loss_train ,num_train = [],[]
num=0
model.train()
if __name__ == '__main__':
    train_data = train_Dataset('./titanic/train.csv')
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=128,  # 每一个batch的样本量 batch数量是iterable
        shuffle=True,
        num_workers=8  # 多线程数
    )
    for epoch in range(60):
        sum_loss = 0.0
        for i,data in enumerate(train_loader,0):
            inputs , labels = data

            y_pred = model(inputs)
            # print(y_pred)
            loss = criterion(y_pred,labels)
            sum_loss += loss.item()
            print('epoch={},i={},loss={}'.format(epoch,i,loss.item()))
            optimizer.zero_grad() #在反向传播之前把所有权重的梯度归零
            loss.backward() #反向传播
            optimizer.step() #更新
        print("sum_loss:",sum_loss / 891)
        sum_loss=sum_loss / 891
        loss_train.append(sum_loss)
        num += 1
        num_train.append(num)
    state = {'model':model.state_dict()}
    torch.save(state, 'titanic_model.pth')

    plt.figure(figsize=(6, 3))
    plt.title('Loss-num_train', fontsize=22)
    plt.xlabel('num_train', fontsize=22)
    plt.ylabel('Loss', fontsize=22)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.plot(num_train, loss_train, 'r-')
    filePath = '60_loss.png'
    plt.savefig(filePath)
    plt.gcf().clear()

    #测试集
    test_data = test_dataset('./titanic/test.csv')
    y_test_pred ,PassengerId = [],[]
    for i in range(test_data.length()):
        x_test , passengerid = test_data.test_sample(i)
        PassengerId.append(passengerid)
        y_p = model(torch.Tensor(x_test.reshape(1,-1)))
        y_p = y_p.item()
        if y_p < 0.5:
            y_p = 0
        else:
            y_p = 1
        y_test_pred.append(y_p)
    result = pd.DataFrame({'PassengerId': PassengerId, 'Survived': y_test_pred})
    result.to_csv("logistic_regression_predictions.csv", index=False)











