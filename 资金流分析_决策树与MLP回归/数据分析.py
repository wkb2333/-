import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing as pc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_curve, auc
from sklearn import metrics, tree
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import GridSearchCV
import graphviz
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt


def plot_roc(y_true, y_score):
    plt.figure(figsize=(5, 5))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.savefig('可视化/决策树分类roc.png')
    plt.show()


# 数据处理
df1 = pd.read_csv('东方财富个股资金流.csv')
# 删除无数值意义的数据
df1.drop(columns=['股票代码', '股票名称'], inplace=True)
k1 = df1.isnull().sum()
k1.sort_values(ascending=False, inplace=True)
print(k1)
# 发现没有缺失值，不需要额外处理

# 对数据进行规范化，标准化化为均值为0，标准差为1的数据
nm = pc.scale(df1)
df_nm = pd.DataFrame(nm, columns=df1.columns)

# 划分测试集、训练集
col = ['最新价', '今日主力净流入净额', '今日主力净流入净占比', '今日超大单净流入净额', '今日超大单净流入净占比',
       '今日大单净流入净额', '今日大单净流入净占比', '今日中单净流入净额', '今日中单净流入净占比', '今日小单净流入净额',
       '今日小单净流入净占比']
x = df_nm[col]
y = df_nm['今日涨跌额']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8)

# 数据分析
# 今日涨跌额与其他数据的spearman相关系数
tmk = df_nm.corr(method='spearman')
stmk = tmk['今日涨跌额']
# print(stmk.sort_values(ascending=False))


rc = {'font.sans-serif': 'SimHei',
      'axes.unicode_minus': False}
sns.set(font_scale=0.9,rc=rc)

plt.bar(stmk.keys(), stmk.values)
plt.xticks(rotation=90)
plt.title('spearman相关系数')
plt.ylabel('相关系数')
plt.show()

# 绘出pearson相关系数热力图
cor = df_nm.corr('pearson')

plt.title('pearson相关系数热力图')
sns.heatmap(cor,
            annot=True,  # 显示相关系数的数据
            center=0.5,  # 居中
            fmt='.2f',  # 只显示两位小数
            vmin=0, vmax=1,  # 设置数值最小值和最大值
            xticklabels=True, yticklabels=True,  # 显示x轴和y轴
            square=True,  # 每个方格都是正方形
            cbar=True,  # 绘制颜色条
            )
plt.savefig("可视化/皮尔逊相关系数热力图.png")
plt.show()

# 数据挖掘
# PCA

# 使用PCA将12个维度的数据压缩到两个维度上，绘出散点图
pca2d = PCA(n_components=2)
pca2d.fit(df_nm)
x = pca2d.transform(df_nm)
plt.title('PCA二维图像')
plt.scatter(x[:, 0], x[:, 1], marker='+', alpha=0.5)
plt.show()

print(pca2d.explained_variance_ratio_, pca2d.explained_variance_ratio_.sum())
print(pca2d.explained_variance_)

# 经测试，5维条件下所含信息能够达到0.85
pca = PCA(n_components=5)
pca.fit(df_nm)
x = pca.transform(df_nm)
print(pca.explained_variance_ratio_, pca.explained_variance_ratio_.sum())
print(pca.explained_variance_)

# FFN回归预测今日涨跌额

class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(11, 128),
            nn.Sigmoid(),
            nn.Linear(128, 1)            
        )
    
    def forward(self, x):
        return self.ffn(x)

xt_train = torch.from_numpy(x_train.to_numpy()).to(torch.float32)
xt_test = torch.from_numpy(x_test.to_numpy()).to(torch.float32)
yt_train = torch.from_numpy(y_train.to_numpy()).to(torch.float32).reshape(-1, 1)
yt_test = torch.from_numpy(y_test.to_numpy()).to(torch.float32).reshape(-1, 1)

loss_train = []
loss_valid = []

net = FFN()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
loss_func = nn.MSELoss()
for t in range(2000):
    pred = net(xt_train)
    loss = loss_func(pred, yt_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        loss_train.append(mean_squared_error(yt_train, pred))
        loss_valid.append(mean_squared_error(yt_test, net(xt_test)))

plt.plot(loss_train, label='loss_train')
plt.plot(loss_valid, label='loss_valid')
plt.xlabel('iters')
plt.ylabel('loss')
plt.title('mean squared error loss')
plt.legend()
plt.show()

plt.plot(loss_train[25:], label='loss_train')
plt.plot(loss_valid[25:], label='loss_valid')
plt.xlabel('iters')
plt.ylabel('loss')
plt.title('mean squared error loss')
plt.legend()
plt.show()

with torch.no_grad():
    plt.title('train')
    plt.plot(net(xt_train), alpha=0.7)
    plt.plot(yt_train, alpha=0.4)
    plt.show()
    
    plt.title('valid')
    plt.plot(net(xt_test), alpha=0.5)
    plt.plot(yt_test, alpha=0.5)
    plt.show()

# 决策树分类今日涨跌情况

y_train = y_train.map(lambda x: 0 if x < 0 else 1)
y_test = y_test.map(lambda x: 0 if x < 0 else 1)


parameters = {
    'splitter': ('best', 'random'),
    'criterion': ('gini', 'entropy'),
    'max_depth': [*range(1, 10)]
}
dt = DT()
gs = GridSearchCV(dt, parameters, cv=5)
gs.fit(x_train, y_train)
b_param = gs.best_params_
b_score = gs.best_score_
print('网格搜索最佳参数设置：', b_param)
print('网格搜索最佳得分：', b_score)
clf = DT(
    splitter=b_param['splitter'],
    criterion=b_param['criterion'],
    max_depth=b_param['max_depth']
)
clf.fit(x_train, y_train)
pred_y = clf.predict(x_test)
print('网格搜索决策树最佳得分', clf.score(x_test, y_test))
print('混淆矩阵：\n', confusion_matrix(y_test, pred_y))
print('accuracy score = {:.2f}'.format(metrics.accuracy_score(y_test, pred_y)))
print('f1 score = {:.2f}'.format(metrics.f1_score(y_test, pred_y)))
print('precision score = {:.2f}'.format(metrics.precision_score(y_test, pred_y)))
print('recall score = {:.2f}'.format(metrics.recall_score(y_test, pred_y)))
print('roc auc score = {:.2f}'.format(metrics.roc_auc_score(y_test, pred_y)))
plot_roc(y_test, pred_y)

dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    feature_names=col,
    class_names=['0', '1'],
    filled=True,
    rounded=True
)
graph = graphviz.Source(dot_data, encoding='UTF-8')
graph.view('可视化/决策树可视化')
