import numpy as np
import pandas as pd
from _collections import deque
import random
from math import log

#CART
class BTree(object):

    # 初始化
    def __init__(self, data=None, left=None, right=None,features=None,epi=0.01,i=None):
        self.data = data    # 所包含的行
        self.left = left    # 左子树 表示是
        self.right = right  # 右子树 表示否
        self.features=features #包含的特征类
        self.ninc=[] #不包含的特征值,降维表示，如[5,1,3,1]表示特征5不包含1 特征3不包含1
        self.epi=epi
        self.select=None#选取的特征
        self.selectnum=None
        self.i=i #标签对应值

CartTree=BTree()
que_bfs=deque()

def calc_gini(data,feat,i):#data为包含行,feat为特征,i为特征对应值
    tot_neg=0
    tot_target_correct=0#正类的label=1
    tot_target_correct2=0#负类的label=1
    res=0
    for lines in data:
        if(datasets[lines][feat] == i):#正类
            if(datasets[lines][-1] == 1):
                tot_target_correct+=1

        else:#负类
            tot_neg+=1
            if (datasets[lines][-1] == 1):
                tot_target_correct2 += 1
    tot_pos=len(data)-tot_neg#正类
    if tot_pos !=0:
        res=(tot_pos*2*tot_target_correct/tot_pos*(1-((tot_target_correct)/tot_pos)))/len(data)
    if tot_neg !=0:
        res+=(tot_neg*2*tot_target_correct2/tot_neg*(1-((tot_target_correct2)/tot_neg)))/len(data)#算基尼
    return res

def calc_min_gini(tree):
    gini=999
    cur_feat=999#当前最优特征
    cur_i=999#最优的取值
    for feat in tree.features:
        for i in diff_count[feat]:
            flag = 0  # 去重标志
            if len(tree.ninc)>0:
                for p in range(int(len(tree.ninc)/2)):
                    if tree.ninc[p*2]==feat and tree.ninc[p*2+1]==i:
                        flag=1
                        break
            if(flag):
                continue
            a=calc_gini(tree.data,feat,i)
            if a < gini:
                gini=a
                cur_feat=feat
                cur_i=i
    return gini,cur_feat,cur_i#返回最小基尼和最小特征值

def update_data(parent_tree,feature_name,i,is_pos=1): #is_pos=1表示正类 =0表示负类 i表示对应的值
    p=list(parent_tree.data)
    for datas in parent_tree.data:
        if(datasets[datas][feature_name] == i):
            if not (is_pos):
                p.remove(datas)
        else:
            if is_pos:
                p.remove(datas)
    return p


def build_mytree(tree):
    if len(tree.data)==0 or len(tree.features)==0:
        return 0
    global que_bfs
    min_gini,feat,i=calc_min_gini(tree)#计算最小基尼和对应特征
    if(min_gini > tree.epi):
        tree.select = feat
        tree.selectnum=i
        p=list(tree.features)
        p.remove(feat)
        tree_left=BTree(features=p)
        tree_left.data=update_data(tree,feat,i,1)#更新特征
        tree_left.i=i
        tree_left.ninc = list(tree.ninc)
        tree_right=BTree(features=tree.features)
        tree_right.data=update_data(tree,feat,i,0)#更新特征
        tree_right.ninc=list(tree.ninc)
        tree_right.ninc.append(feat)
        tree_right.ninc.append(i)#不包含的特征
        tree_right.i = i
        tree.left=tree_left
        tree.right=tree_right
        que_bfs.append(tree_left)
        que_bfs.append(tree_right)


def count_dif(ds):#统计各行不同的种类
    nn=[]
    i=0
    for col in ds.T:
        a = []
        for nums in col:
            if not nums in a:
                a.append(nums)
        nn.append(a)
        i+=1
    return nn

def predict_data(data,tree):#预测,最后返回一棵树
    if(tree.select==None):
        return tree
    if(datasets[data][tree.select]==tree.selectnum):
        if (len(tree.left.data) == 0):
            return tree
        return predict_data(data,tree.left)
    else:
        if (len(tree.right.data) == 0):
            return tree
        return predict_data(data,tree.right)

def count_most(dt):
    pos,neg=0,0
    for d in dt:
        if datasets[d][-1]==1:
            pos+=1
        else:
            neg+=1
    if(pos >= neg):
        return 1
    else:
        return 0

df = pd.read_excel(r"DS.xls")
#获取数据集和每个维度的名称
df = df.drop(['nameid'], axis=1) #把第 1 列 ID 数据去除掉
re = [0,10000,20000,30000,40000,50000]
df['revenue']=pd.cut(df['revenue'],re,labels=False)
datasets = df.values
diff_count=count_dif(datasets)#统计不同取值
labels = df.columns.values
list_all=[i for i in range(datasets.shape[0])]
list_test=random.sample(list_all,200)
testdata=[]
CartTree.data=[]
tp,fp,tn,fn=0,0,0,0
for j in list_all:
    if not j in list_test:
        CartTree.data.append(j)
    else:
        testdata.append(j)
CartTree.features=[i for i in range(len(diff_count)-1)]
que_bfs.append(CartTree)

while que_bfs:
    tree_cur = que_bfs.popleft()
    build_mytree(tree_cur)

#预测
for dt in testdata:
    tree_pre=predict_data(dt,CartTree)
    ans=count_most(tree_pre.data)
    #统计树中的多数
    if ans==1 and datasets[dt][-1]==1:
        tp+=1
    if ans==1 and datasets[dt][-1]==0:
        fp+=1
    if ans==0 and datasets[dt][-1]==1:
        fn+=1
    if ans==0 and datasets[dt][-1]==0:
        tn+=1

print("Precision:"+str(tp/(tp+fp)))
print("Recall:"+str((tp)/(tp+fn)))
print("F1Score:"+str(2*tp/(2*tp+fp+fn)))
