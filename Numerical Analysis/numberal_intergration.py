import numpy as np
import matplotlib.pyplot as plt
import math

def f_1(x):
    return 1/(1+x**2)
def linspace(a,b,n):
    return np.linspace(a, b, n)
def func_trapezoid(points,i,basefunc=f_1):
    return (points[i+1]-points[i])*(basefunc(points[i])+basefunc(points[i+1]))/2
def func_simpsons(points,i,basefunc=f_1):
    return (points[i+1]-points[i])*(basefunc(points[i])+4*basefunc((points[i]+points[i+1])/2)+basefunc(points[i+1]))/6
def func_gaussian(points,i,num=3,basefunc=f_1):
    w,a,b=gen_ort_poly(points[i],points[i+1],num)
    eig,feat=get_zero_point(a,b)#eig=zerospoints
    A=get_param_a(eig,points[i],points[i+1])
    res=0
    for i in range(len(A)):
        res+=A[i]*basefunc(eig[i])
    return res
def calc_poly(poly,n):
    ans=0
    for i in range(len(poly)):
        ans+=poly[i]*n**(len(poly)-i-1)
    return ans
def poly_intergrate(poly,a,b):
    new_r=[0]*(len(poly)+1)
    for i in range(len(poly)):
        new_r[i]=poly[i]/(len(poly)-i)
    return calc_poly(new_r,b)-calc_poly(new_r,a)
def poly_multiply(a,b):
    res=[0]*(len(a)+len(b)-1)
    for i in range(len(a)):
        for j in range(len(b)):
            res[i+j]+=a[i]*b[j]
    return res
def poly_add(a,b):
    c=list(a)
    d=list(b)
    if(len(c)<len(d)):
        e=list(d)
        d=list(c)
        c=list(e)

    res=list(a)
    for i in range(len(d)):
        res[i+len(c)-len(d)]+=d[i]
    return res

def gen_ort_poly(a,b,num):#by recursion ,num=1 returns 1
    w0=[0]
    w1=[1]
    w2=[1]
    bn = 0
    alpha=[0.]*(num-1)
    beta=[0.]*(num-1)
    for j in range(num-1):
        poly_w_w=poly_multiply(w1,w1)
        poly_w_w2=poly_multiply(w0,w0)
        poly_x_w2=poly_multiply([1,0],poly_w_w)
        an=poly_intergrate(poly_x_w2,a,b)/poly_intergrate(poly_w_w,a,b)
        alpha[j]=an
        if(poly_intergrate(poly_w_w2,a,b)!=0):
            bn=poly_intergrate(poly_w_w,a,b)/poly_intergrate(poly_w_w2,a,b)
            beta[j]=bn
        w2=[0]*(len(w1)+1)
        w2=poly_add(w2,poly_multiply([1,0],w1))
        w_t1=poly_multiply([-an],w1)
        w_t2=poly_multiply([-bn],w0)
        w2=poly_add(w2,w_t1)
        w2=poly_add(w2,w_t2)
        w0=list(w1)
        w1 = list(w2)
    return w2,alpha,beta

def get_zero_point(alpha,beta):
    Martrix=np.zeros((len(alpha),len(alpha)))
    for i in range(len(alpha)):
        Martrix[i][i]=alpha[i]
        if(i!=len(alpha)-1):
            Martrix[i][i+1]=1
            Martrix[i+1][i]=beta[i+1]

    eig,featvec=np.linalg.eig(Martrix)
    return eig,featvec

def Gauss_GenLinar(zp,a,b):
    Mat=np.zeros((len(zp),len(zp)))
    Vec=np.zeros((len(zp),1))
    aa=[1]
    for i in range(len(zp)):
        f=list(aa)
        f_inter=poly_intergrate(f,a,b)
        for zeropoint in range(len(zp)):
            Mat[i][zeropoint]=calc_poly(f,zp[zeropoint])
        Vec[i][0]=f_inter
        aa.append(0)
    return Mat,Vec

def get_param_a(zeropoints,a,b):
    Mat,Vec=Gauss_GenLinar(zeropoints,a,b)
    A=np.linalg.solve(Mat,Vec)
    return A



def count_allseg(points,func,num=0):
    res=0.
    for i in range(len(points)-1):
        if(num!=0):
            res+=func(points,i,num)
        else:
            res+=func(points,i)
    print("res:"+str(res))
    bias=math.fabs(math.pi/2 - res)
    print("bias:"+str(bias))

def solve_q1():
    p=linspace(-1,1,20)
    count_allseg(p,func_trapezoid)

def solve_q2():
    p=linspace(-1,1,10)
    count_allseg(p,func_simpsons)
def solve_q3():
    p=linspace(-1,1,10)
    count_allseg(p,func_gaussian,num=3)
def solve_q4():
    p=linspace(-1,1,6)
    count_allseg(p,func_gaussian,num=4)
def solve_q5():
    p=linspace(-1,1,4)
    count_allseg(p,func_gaussian,num=6)



#solve_q1()
#solve_q2()
#solve_q3()
#solve_q4()
solve_q5()