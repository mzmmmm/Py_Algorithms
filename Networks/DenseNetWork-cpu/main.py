# This is a sample Python script.
import numpy as np
from DataSet import load_mnist
import random
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def NormalDist(loc=0,scale=1,size=None):
    return np.random.normal(loc,scale,size)
def SizeDist(f,size,*args):
    return f(*args,size=size)
def ReLu(x):
    y=x
    for j in range(y.shape[0]):
        for i in range(y.shape[1]):
            if( y[j,i]<= 0):
                y[j,i]=0
    return y
def Identi(x):
    return x

idx=0
NeruNodeSets=[]
class NeruNode(object):
    def __init__(self,activation,):
        self.prev_node=[]
        self.prev_weight=[]
        self.activation=activation
        NeruNodeSets.append(self)
        global idx
        idx = idx + 1
    def AppendDenseLayerNodes(self,layer,initialize=NormalDist,*args):
        for nodes in layer.NeruNode:
            self.prev_node.append(nodes)
            self.prev_weight=np.append(self.prev_weight,initialize(*args))
class InputMatrix(object):
    def __init__(self,size): # size should be a tuple
        self.Matrix=np.zeros(size)
    def ConvertToMatrix(self,*args):
        raise NotImplementedError

class Layers(object):
    def __init__(self,prev_layer=None,next_layer=None,num=10,activation=ReLu):
        self.is_output=False
        self.is_firstlayer=False
        self.nextlayer=None
        self.prevlayer=None
        self.num=num #Neru num
        self.activation=activation # actv func , relu by default
        self.CurrentWeight=[]
        self.CurrentValue=[]
        self.CurrentBias=[]
        if (prev_layer == None):
            self.is_firstlayer = True
        else:
            self.prevlayer = prev_layer
            prev_layer.nextlayer=self
            prev_layer.__update_layer_status()
            self.__update_layer_status()
        if (next_layer == None):
            self.is_output = True
        else:
            self.nextlayer = next_layer
            next_layer.prevlayer=self
            next_layer.__update_layer_status()
            self.__update_layer_status()
    def __update_layer_status(self):
        if (self.prevlayer == None):
            self.is_firstlayer = True
        else:
            self.is_firstlayer = False
        if (self.nextlayer == None):
            self.is_output = True
        else:
            self.is_output=False
    def SetNextLayer(self,nextlay):
        self.nextlayer=nextlay
        nextlay.prevlayer=self
        nextlay.__update_layer_status()
        self.__update_layer_status()
    def SetPrevLayer(self,nextlay):
        self.prevlayer=nextlay
        nextlay.nextlayer=self
        nextlay.__update_layer_status()
        self.__update_layer_status()

class DenseLayers(Layers):
    def __init__(self,prev_layer=None,next_layer=None,num=10,init_method=NormalDist,*args,activation=ReLu):
        super(DenseLayers,self).__init__(prev_layer,next_layer,num,activation=activation)
        self.Output=None
        if(prev_layer!=None):
            self.CurrentValue=np.asmatrix(np.zeros(self.num)).transpose()
            Size=(self.num,prev_layer.num)
            self.CurrentWeight=np.asmatrix(SizeDist(init_method,Size,*args))
            self.CurrentBias=np.asmatrix(np.zeros(self.num)).transpose()
    def Update(self,delta,biasup,lr=0.001): # delta must be the same size matrix, used only in the last layer
        i=0
        lay=self
        while(lay.prevlayer != None and not lay.is_firstlayer):
            try:
                lay.CurrentWeight = lay.CurrentWeight - delta[i]*lr
            except:
                print("err")
                return
            lay.CurrentBias = lay.CurrentBias - biasup[i] * lr
            lay=lay.prevlayer
            i = i + 1
    def SetOutPut(self,output="softmax"): # output is kind of string
        if(output == "Linear"):
            self.Output=None
        else:
            self.Output=output


def ForwardBrocast(InputVector,InputLayer):
    lay = InputLayer  # its a pointer
    multivec=InputVector
    multivec=np.asmatrix(multivec).transpose()
    while lay != None:
        if(lay.is_firstlayer):
            lay.CurrentValue=multivec
            lay=lay.nextlayer
            continue
        lay.CurrentValue = lay.CurrentWeight @ multivec + lay.CurrentBias
        if(lay.activation!= None):
            lay.CurrentValue = lay.activation(lay.CurrentValue)
        multivec=lay.CurrentValue
        lay=lay.nextlayer

def modify(x,pos,ele):
    if(len(x) < pos):
        x.append(ele)
    else:
        x[pos-1]= x[pos-1] + ele
def calcgrad(w1,mats,laynum,lay,multiplyer=None):
    p=[]
    bs=[]
    if(multiplyer==None):
        multiplyer=np.asmatrix([1]*w1.shape[0])

    if(laynum==2):

        for i in range(lay.num):
            bs.append((multiplyer @ w1[:,i])[0,0])
            p2=[]
            for j in range(lay.prevlayer.num):
                p2.append((multiplyer@w1[:,i]*lay.prevlayer.CurrentValue[j,0])[0,0])
            p.append(p2)
    else:
        p2=(multiplyer @ w1 @ mats[laynum - 3])
        for i in range(lay.num):
            bs.append(p2[0,i])
            p3=[]
            for j in range(lay.prevlayer.num):
                p3.append(p2[0,i] * lay.prevlayer.CurrentValue[j,0])
            p.append(p3)
    return np.asmatrix(p),np.asmatrix(bs).transpose()

def modify_outputs(layer):
    if layer.Output==None:
        return layer.CurrentValue
    if layer.Output=="Softmax":
        returns=np.zeros(shape=(layer.num,1))
        max_z=np.max(layer.CurrentValue)# to avoid overflow
        sums=np.ones_like([1]*layer.num) @ np.exp(layer.CurrentValue-max_z)
        sums=sums[0,0]

        for i in range(layer.num):
            returns[i,0] = np.exp(layer.CurrentValue[i,0]-max_z)/sums
        return np.asmatrix(returns)
    return None

def GetMultiplyer(layer,id,modi):
    if(layer.Output==None):
        return [1]*layer.num
    if(layer.Output=="Softmax"):
        rt=[]
        for i in range(layer.num):
            if(i==id):
                rt.append(modi[id,0]*(1-modi[id,0]))
            else:
                rt.append(-modi[id,0]*modi[i,0])
        return rt
def BackPropagation(LastLayer,Correct_Ans):
    id = 0
    mats=[] # matrix for derivative
    delta_applied =[] #matrix for updating
    bias_applied=[]
    offsets=[]
    # last layer
    total_lost=0
    total_lost_raw=0
    modified_value=modify_outputs(LastLayer)

    for outputs in modified_value:

        w1=LastLayer.CurrentWeight
        raw_offs=LastLayer.CurrentValue[id,0]- Correct_Ans[id]
        offset=outputs[0,0] - Correct_Ans[id]
        total_lost = total_lost + abs(offset)
        total_lost_raw = total_lost_raw + abs(raw_offs)
        offsets.append(offset)
        multiplyer=GetMultiplyer(LastLayer,id,modified_value)
        modify(bias_applied, 1, np.diag(multiplyer) @ np.asmatrix(np.ones((1, LastLayer.num))).transpose() * offset )
        if(id ==0):
            lay=LastLayer.prevlayer # pointer
            if(lay != None):
                mats.append(lay.CurrentWeight)
                lay=lay.prevlayer
                while lay!= None and not lay.is_firstlayer :
                    mats.append(mats[-1]@lay.CurrentWeight)
                    lay = lay.prevlayer
            if (offset == 0 or multiplyer[id] == 0.):
                delta_applied.append(np.asmatrix(np.zeros((1, LastLayer.prevlayer.num))))
                id = id + 1
                continue
            modify(delta_applied, 1, LastLayer.prevlayer.CurrentValue.transpose()*offset*multiplyer[id])

        else:
            if (offset == 0 or multiplyer[id] == 0.):
                delta_applied[0] = np.row_stack((delta_applied[0], np.zeros((1, LastLayer.prevlayer.num))))
                id = id + 1
                continue
            delta_applied[0]=np.row_stack((delta_applied[0],LastLayer.prevlayer.CurrentValue.transpose()*offset*multiplyer[id]))


        lay =LastLayer
        lay=lay.prevlayer
        laynum=2
        while(lay != None):
            if(lay.prevlayer != None):
                grad,bias=calcgrad(w1,mats,laynum,lay,multiplyer)
                modify(delta_applied,laynum,grad*offset)
                modify(bias_applied,laynum,bias*offset)
            lay=lay.prevlayer
            laynum = laynum + 1
        id = id + 1
    if(len(delta_applied) <3): # no need to update , all correct
        return 0,0,total_lost,offsets,total_lost_raw
    return delta_applied,bias_applied,total_lost,offsets,total_lost_raw
def ModelTest():
    input=Layers(num=784,activation=Identi)
    dense1=DenseLayers(input,num=32)
    dense2=DenseLayers(dense1,num=16)#only 3 layers have issue
    dense3=DenseLayers(dense2,num=16)#only 3 layers have issue
    output=DenseLayers(dense3,num=10,activation=Identi)
    output.SetOutPut("Linear")
    epoch=100000
    batchsize=30
    (x, y) = load_data_mnist()
    for i in range(epoch):
        ts_epoch=0
        tsr_epoch=0
        randomlist = random.sample(range(0,len(x[0])), batchsize)
        x_t=[x[0][i] for i in randomlist]
        x_label=[x[1][i] for i in randomlist]
        for j in range(batchsize):
            ForwardBrocast(x_t[j],input)
            ds,bs,ts,offs,rts=BackPropagation(output,x_label[j])
            ts_epoch = ts_epoch + ts
            tsr_epoch = tsr_epoch + rts
            if(ds == 0):
                #print("encounter 0")
                continue
            with open("a.txt","a+") as f:
                f.write(str(ds) + "\n"+str(bs) + "\n\n")
                pass
            output.Update(ds,bs,lr=0.000001) # if last layer is a simple identi map layer , then lr must be manually modified
        print("total score: "+ str(ts_epoch))
        print("total score raw: "+ str(tsr_epoch))


def load_data_mnist():
    x_train_path='./Mnist/train-images-idx3-ubyte.gz'
    y_train_path='./Mnist/train-labels-idx1-ubyte.gz'
    x_test_path='./Mnist/t10k-images-idx3-ubyte.gz'
    y_test_path='./Mnist/t10k-labels-idx1-ubyte.gz'
    return load_mnist(x_train_path, y_train_path, x_test_path, y_test_path)

ModelTest()