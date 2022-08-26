import pandas as pd
import numpy as np
import torch
import re
import gc
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

#仅提取数字部分，也就是把年数拿出来，<1的年数变为0
def workYearDIc(x):
    if str(x) == 'nan':
        return 0
    x = x.replace('< 1', '0')#字符串替换
    return int(re.search('(\d+)', x).group())

def findDig(val):
    fd = re.search('(\d+-)', val)
    if fd is None:
        return '1-' + val
    return val + '-01'


# 定义文件路径
train_internet_path = "./train_internet.csv"
train_public_path = "./train_public.csv"
test_public_path = "./test_public.csv"

#读取文件
train_internet_data = pd.read_csv(train_internet_path)
train_public_data = pd.read_csv(train_public_path)
test_public_data = pd.read_csv(test_public_path)

#获取列名称
train_internet_list = list(train_internet_data.columns)
train_public_list = list(train_public_data.columns)
test_public_list = list(test_public_data.columns)

#获取列的个数
train_internet_list_count = len(train_internet_list)
train_public_list_count = len(train_public_list)
test_public_list_count = len(test_public_list)

#把数值列的名称转换为一个列表
train_internet_data_num_columns = train_internet_data.select_dtypes(exclude="object").columns.tolist()
train_public_data_num_columns = train_public_data.select_dtypes(exclude="object").columns.tolist()
test_public_data_num_columns = test_public_data.select_dtypes(exclude='object').columns.tolist()

'''处理train_internet数据'''

#查找包含NAN的列的名称，通过字典提取出来为一个列表
train_internet_dict = train_internet_data.isnull().any().to_dict()
train_internet_dict.pop("post_code")
train_internet_data_columns_isnull = train_internet_dict.keys()

# 用中位数填充包含NAN的数值列
train_internet_data[train_internet_data_columns_isnull].select_dtypes(exclude='object').\
    fillna(train_internet_data[train_internet_data_columns_isnull].select_dtypes(exclude='object').median())

#给post_code列填充NAN，post_code是编号，而不是数值类型的数据，因此给NAN赋值列中出现次数最多的编号
train_internet_data['post_code'].fillna(train_internet_data['post_code'].mode()[0])

#将存在负数的列中负数变为0
for i in range(len(train_internet_data['debt_loan_ratio'])):
    if train_internet_data['debt_loan_ratio'][i]<0:
        train_internet_data['debt_loan_ratio'][i]=0

# 不知道干啥的
f_feas = ['f0', 'f1', 'f2', 'f3', 'f4']
for f in f_feas:
    train_internet_data[f'industry_to_mean_{f}'] = train_internet_data.groupby('industry')[f].transform('mean')

'''处理train_public数据'''

#查找包含NAN的列的名称
train_public_dict = train_public_data.isnull().any().to_dict()
del train_public_dict['post_code']
train_public_data_columns_isnull = train_public_dict.keys()
train_public_data_columns_isnull_num = list(train_public_data[train_public_data_columns_isnull].select_dtypes(exclude='object'))

# 用中位数填充包含NAN的数值列
train_public_data[train_public_data_columns_isnull_num]= \
    train_public_data[train_public_data_columns_isnull_num].\
        fillna(train_public_data[train_public_data_columns_isnull_num].median())


#给post_code列填充出现最多次的数据
train_public_data['post_code'].fillna(train_public_data['post_code'].mode()[0])

#将存在负数的列中负数变为0
for i in range(len(train_public_data['debt_loan_ratio'])):
    if train_public_data['debt_loan_ratio'][i]<0:
        train_public_data['debt_loan_ratio'][i]=0

for f in f_feas:
    train_public_data[f'industry_to_mean_{f}'] = train_public_data.groupby('industry')[f].transform('mean')

'''处理test_public数据'''
#查找包含NAN的列的名称
test_public_dict = test_public_data.isnull().any().to_dict()
del test_public_dict['post_code']
test_public_data_columns_isnull = test_public_dict.keys()
test_public_data_columns_isnull_num = list(test_public_data[test_public_data_columns_isnull].select_dtypes(exclude='object'))

#用中位数填充NAN
test_public_data[test_public_data_columns_isnull_num]=\
test_public_data[test_public_data_columns_isnull_num].\
    fillna(test_public_data[test_public_data_columns_isnull_num].median())

#用出现最多的数据填充post_code
test_public_data['post_code'].fillna(test_public_data['post_code'].mode()[0])

for f in f_feas:
    test_public_data[f'industry_to_mean_{f}'] = test_public_data.groupby('industry')[f].transform('mean')

# print(train_public_data['policy_code'].unique())
# pllicy_code这一列只有1这个数值，应当丢弃
'''数据处理完成'''

# 把work_year这一列的数字提取出来，不够一年的按0算，并转换为int64类型
timeMax = pd.to_datetime('1-Dec-21')
train_internet_data['work_year'] \
= train_internet_data['work_year'].map(workYearDIc)

train_public_data['work_year']\
=train_public_data['work_year'].map(workYearDIc)

test_public_data['work_year']\
=test_public_data['work_year'].map(workYearDIc)

# 把'class'这一列转换为数值类型
class_dict = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
}

train_internet_data['class'] \
= train_internet_data['class'].map(class_dict) #也可以用.replace()方法

train_public_data['class'] \
= train_public_data['class'].map(class_dict)

test_public_data['class']\
= test_public_data['class'].map(class_dict)

#把May-84这种月份加年份的形式，加一个1号，形成一个完整日期
#把这个数据列转换为日期形式
train_internet_data['earlies_credit_mon']=\
pd.to_datetime(train_internet_data['earlies_credit_mon'].map(findDig))

train_public_data['earlies_credit_mon']=\
pd.to_datetime(train_public_data['earlies_credit_mon'].map(findDig))

test_public_data['earlies_credit_mon']=\
pd.to_datetime(test_public_data['earlies_credit_mon'].map(findDig))

# 超过设定上限timeMax的时间，年份减去100
# internet数据直接转换成为标准日期形式，不做-100处理
train_internet_data['earlies_credit_mon'] = pd.to_datetime(train_internet_data['earlies_credit_mon'])
train_public_data.loc[train_public_data['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] \
    = train_public_data.loc[train_public_data['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(years=-100)
test_public_data.loc[test_public_data['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] \
    = test_public_data.loc[test_public_data['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(years=-100)

#把issue_date转换为年月日的表示
train_internet_data['issue_date'] = \
pd.to_datetime(train_internet_data['issue_date'])

train_public_data['issue_date'] = \
pd.to_datetime(train_public_data['issue_date'])

test_public_data['issue_date'] = \
pd.to_datetime(test_public_data['issue_date'])

'''添加三列issue_date_month,
issue_date_year,issue_date_dayofweek数值形式，表示月份和年份，以及一周中的第几天'''
train_internet_data['issue_date_month'] = train_internet_data['issue_date'].dt.month
train_internet_data['issue_date_year'] = train_internet_data['issue_date'].dt.year
train_internet_data['issue_date_dayofweek'] = train_internet_data['issue_date'].dt.dayofweek

train_public_data['issue_date_month'] = train_public_data['issue_date'].dt.month
train_public_data['issue_date_year'] = train_public_data['issue_date'].dt.year
train_public_data['issue_date_dayofweek'] = train_public_data['issue_date'].dt.dayofweek

test_public_data['issue_date_month'] = test_public_data['issue_date'].dt.month
test_public_data['issue_date_year'] = test_public_data['issue_date'].dt.year
test_public_data['issue_date_dayofweek'] = test_public_data['issue_date'].dt.dayofweek

'''添加两列，表示earlies Credit的月份和年份'''
train_internet_data['earliesCreditMon'] = train_internet_data['earlies_credit_mon'].dt.month
train_internet_data['earliesCreditYear'] = train_internet_data['earlies_credit_mon'].dt.year

train_public_data['earliesCreditMon'] = train_public_data['earlies_credit_mon'].dt.month
train_public_data['earliesCreditYear'] = train_public_data['earlies_credit_mon'].dt.year

test_public_data['earliesCreditMon'] = test_public_data['earlies_credit_mon'].dt.month
test_public_data['earliesCreditYear'] = test_public_data['earlies_credit_mon'].dt.year

############################################编码部分
# 给非数值类型列编码
list_encode = ['work_type', 'employer_type', 'industry']

for list_n in list_encode:
    list_m = train_internet_data[list_n].unique().tolist()
    dict_n = dict(zip(list_m, list(range(len(list_m)))))
    train_internet_data[list_n] = train_internet_data[list_n].map(dict_n)

list_encode = ['employer_type', 'industry']
for list_n in list_encode:
    list_m = train_public_data[list_n].unique().tolist()
    dict_n = dict(zip(list_m, list(range(len(list_m)))))
    train_public_data[list_n] = train_public_data[list_n].map(dict_n)

for list_n in list_encode:
    list_m = test_public_data[list_n].unique().tolist()
    dict_n = dict(zip(list_m, list(range(len(list_m)))))
    test_public_data[list_n] = test_public_data[list_n].map(dict_n)

# 丢弃'issue_date', 'earlies_credit_mon'两列
list_drop = ['issue_date', 'earlies_credit_mon']
train_internet_data = train_internet_data.drop(list_drop, axis=1)
train_public_data = train_public_data.drop(list_drop, axis=1)
test_public_data = test_public_data.drop(list_drop, axis=1)

#丢弃internet的sub_class这一列
train_internet_data = train_internet_data.drop('sub_class', axis=1)
#丢弃userid,loanid,policy_code
train_public_data = train_public_data.drop(['user_id', 'loan_id', 'policy_code'], axis=1)
test_public_data = test_public_data.drop(['user_id', 'loan_id', 'policy_code'], axis=1)

# train_data =  pd.read_csv()
#数据处理完成，建立数据类
class Mydataloader(Dataset):
    def __init__(self, dataset, str:str): # dataset是一个dataframe
        super(Mydataloader, self).__init__()
        self.label = dataset[str]
        self.data = dataset.drop([str], axis=1)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        label = torch.tensor(self.label[index])
        data = torch.tensor(self.data.iloc[index])
        data = data.to(torch.float32)

        return data, label

class Testdataloader(Dataset):
    def __init__(self, dataset):
        super(Testdataloader, self).__init__()
        self.data = dataset
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        data = torch.tensor(self.data.iloc[index])
        data = data.to(torch.float32)
        return data

dataset = Mydataloader(train_public_data, 'isDefault')

#训练集大小，80%的数据用来训练
train_size = int(0.8*len(dataset))
#测试集大小，剩余的用来测试
test_size = len(dataset) - train_size
#划分训练集和测试集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
#定义批量大小
batch_size = 16
#定义训练集和测试集的迭代器
train_iter = DataLoader(train_dataset, batch_size=batch_size,
                       shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size,
                       shuffle=True)

#搭建神经网路
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(43, 16*128), nn.ReLU(),
    nn.Linear(16 * 128, 8 * 128), nn.ReLU(),
    nn.Linear(8 * 128, 128), nn.ReLU(),
    nn.Linear(128, 2), nn.Softmax()
)

# 定义损失函数
loss = nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

writer = SummaryWriter('./logs')

# 定义训练步骤和测试步骤
train_step=0
test_step=0
# 开始训练
for epoch in range(5):
    total_accuracy = 0.0
    print(f"----第{epoch+1}轮训练开始----")
    for data in train_iter:
        x,y = data
        y_hat = net(x)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_step+=1
        result_1 = y_hat.argmax(1)
        result = y_hat.argmax(1) == y
        accuracy = result.sum()
        total_accuracy += accuracy
        writer.add_scalar("Train", l, train_step)
        if train_step%100 ==0:
            print(f"第{train_step}次训练，loss：{l}")
    print(f"第{train_step}次训练，准确率：{total_accuracy/train_size}")

    l_sum=0
    #求精准度
    total_accuracy = 0.0
    print(f"----第{epoch+1}轮测试开始----")
    with torch.no_grad():
        for data in test_iter:
            x, y = data
            y_hat = net(x)
            l = loss(y_hat, y)
            l_sum += l
            test_step += 1
            #求精准度
            accuracy = (y_hat.argmax(1) == y).sum()
            writer.add_scalar("Test", l, train_step)
            total_accuracy += accuracy
        print(f"第{epoch+1}轮测试，loss：{l_sum/test_size}, 准确率为：{total_accuracy/test_size}")


# 获取测试数据
test_public_dataset = Testdataloader(test_public_data)
test_public_iter = DataLoader(test_public_dataset, batch_size=batch_size, shuffle=True)
# 进行测试
test_public_step = 0
list_test_isdefault = list(set())
for data in test_public_iter:
    x = data
    y_hat = net(x)
    result = y_hat.argmax(1).tolist()
    list_test_isdefault += result
    test_public_step += 1

# 添加isDefault列，保存为新的csv文件
test_public_data['isDefault'] = pd.Series(list_test_isdefault)
test_public_data.to_csv('./test_public_result.csv')

writer.close()