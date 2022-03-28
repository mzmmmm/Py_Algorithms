import numpy as np
import matplotlib.pyplot as plt
import math
def f_1(x):
    return 1/(1+x**2)
def df_1(x):
    return - 2*x/((1+x**2)**2)

def is_legal(a,length):
    for i in range(len(a)-1):
        if a[i]>=a[i+1] or a[i]>length-1:
            return 0
    if(a[-1] > length-1):
        return 0
    return 1

def count_mount(p,a):#计算组合
    b=1
    pt=0
    for i in range(len(p)):
        if(pt >= len(a)):
            break
        if(a[pt]==i):
            b *= p[i]
            pt+=1
    return b

def get_param_dfs(points,a,num):#num表示轮到第几个a位dfs了
    x=0
    b=list(a)
    b[num - 1] += 1
    if(is_legal(b,len(points))):#位置合法
        x+=count_mount(points,b)
        for i in range(num):
            x+=get_param_dfs(points,b,num-i)
        return x
    return 0


def get_param_langrange(points,num):#(x-a)(x-b)(x-c)，求出x系数的过程
    if(num==0):
        return 1
    a=[i for i in range(num)]
    x=count_mount(points,a)
    x+=get_param_dfs(points,a,num)

    return x
def calc_deno_lag(pt,pts):
    x=1
    for i in pts:
        if(pt==i):
            continue
        x*=(pt-i)
    return x

def lagrange_interpo(points,func):
    x=[0]*len(points)
    #L(x)=x[0]*x^n-1+x[1]*x^n-2...
    for pt in points:
        x_2=[0]*len(points)
        p2=list(points)
        p2.remove(pt)
        denominator = calc_deno_lag(pt,points)#分母
        for g in range(len(p2)):
            p2[g]=-p2[g]

        for i in range(len(points)):
            numerator=get_param_langrange(p2,i)#0 ~ x^n-1 1~x^n-2 ...
            x_2[i]+=numerator/denominator
            x_2[i]*=func(pt)
            x[i]+=x_2[i]
    return x
def gen_herm_polycoeff(points,value):#cal (x-x0)*(x-x1)*...
    x=1
    if(len(points) < 1):
        return 0
    for pt in points:
        x*=(value-pt)
    return x
def calc_coeff_herm(p2x,pt2,points):
    res=[0]*len(p2x)
    #d first
    val_dp2x = d_Lx(p2x)
    val_TT=gen_herm_polycoeff(points,pt2)
    for i in range(len(val_dp2x)):
        res[i]+=val_dp2x[i]*(pt2**(len(val_dp2x)-i-1))*val_TT
    #
    for pts in points:
        points_r=list(points)
        points_r.remove(pts)
        val_TT=gen_herm_polycoeff(points_r,pt2)
        tempo_res=[0]*len(p2x)
        for i in range(len(tempo_res)):
            tempo_res[i]+=pts**((len(tempo_res)-1-i))
        for i in range(len(res)):
            res[i] += tempo_res[i]*val_TT
    return res

def d_Lx(x):
    x2=[0]*(len(x)-1)
    for i in range(len(x)-1):
        x2[i]=x[i]*(len(x)-i-1)
    return x2


def gen_herm_coeff(x_lagrange,points,func,points2): #f'(points2)=func(points2) positive points
    Martrix=np.ones((len(points2),len(points2)))
    Vertex=np.ones((len(points2),1))
    #to solve M*A=V
    dx_lag=d_Lx(x_lagrange)
    p2x=[1]*len(points2)
    val_dL=polygen(dx_lag,points2)
    counter=0
    for pt in points2:
        val_dH=func(pt)
        val_coeff=calc_coeff_herm(p2x,pt,points)#calc coeff of [an,an-1..]
        Vertex[counter][0]=val_dH-val_dL[counter]
        for i in range(len(val_coeff)):
            Martrix[counter][i]=val_coeff[i]
        counter += 1
    Ans=np.linalg.solve(Martrix,Vertex)#get the ans
    #generate the final expression
    exp_right=[0]*(len(points)+1)#calc (x-x0)(x-x1)...
    exp_right2=[0]*(len(points)+len(points2))#calc (an*x+...a0)(x-x0)(x-x1)...
    p2=list(points)
    for i in range(len(points)):
        p2[i]=-points[i]#inverse
    for i in range(len(exp_right)):
        exp_right[i]=get_param_langrange(p2,i)
    for i in range(len(Ans)):
        for j in range(len(exp_right)):
            exp_right2[i+j]+=Ans[i][0]*exp_right[j]
    for i in range(len(x_lagrange)):
        exp_right2[len(exp_right2)-len(x_lagrange)+i]+=x_lagrange[i]
    return exp_right2

def gen_cheby_zeros(num,a=-5,b=5):
    deno=2*num
    x=[0.]*num
    for i in range(num):
        x[i]=math.cos((2*(i+1)-1)/deno*math.pi)*(b-a)/2+(a+b)/2
    return x
def linear_interpo(points,func):
    param=np.zeros((len(points)-1,2)) # A0 k0 A1 k1...
    for i in range(len(points)-1):
        param[i][0]=func(points[i])
        param[i][1]=(func(points[i+1])-func(points[i]))/(points[i+1]-points[i])
        param[i][0]-=param[i][1]*points[i]
    return param
def segmented_hermite_interpo(points):
    param = np.zeros((len(points) - 1, 4))
    for i in range(len(points) - 1):
        l_1=[points[i],points[i+1]]
        x_1 = lagrange_interpo(l_1, f_1)
        param[i] = gen_herm_coeff(x_1, l_1, df_1, l_1)
    return param
def polygen(x,axisx):#x[0]*x^n-1+....
    s=[0]*len(axisx)
    for pt in range(len(axisx)):
        r = 0
        for i in range(len(x)):
            sq=len(x)-1-i
            r+=x[i]*(axisx[pt]**sq)
        s[pt]=r
    return s
def func_polygen(para,x):
    r=0
    for i in range(len(para)):
        sq = len(para) - 1 - i
        r += para[i] * (x ** sq)
    return r
def func_linear(para,x): # line func
    return para[0]+para[1]*x


def segment_gen(x,points,func,para):#segment func , points must be ascending
    i=0
    res=[0]*len(x)
    for p in range(len(x)):
        if x[p] >= points[i] :
            if i < (len(points)-1) and x[p] > points[i+1]:
                i+=1
            res[p]=func(para[i],x[p])

    return res



def funcgen(func,axisx):
    s = [0] * len(axisx)
    for pt in range(len(axisx)):
        s[pt]=func(axisx[pt])
    return s
def count_bias(y1,y2,n):
    p=0.
    for i in range(n):
        p+=math.fabs(y2[i]-y1[i])
    p/=n
    return p
def contra_plot(x_1,func,a=-5.,b=5.,poly=True,points=None,func_seg=None):#point分界线 poly代表为正常模式或者分段模式
    n=101
    x = np.linspace(a, b, n)
    if(poly):
        y = polygen(x_1, x)
    else:
        y=segment_gen(x,points,func_seg,x_1)
    y2 = funcgen(func, x)
    bias=count_bias(y,y2,n)
    plt.plot(x, y, 'r')
    plt.plot(x, y2, 'g')
    print('bias:' + str(bias))
    plt.show()
#三次样条
def tri_h(num,points):# from zero
    return points[num+1]-points[num]
def tri_f(x1,x2,func):
    return (func(x2)-func(x1))/(x2-x1)
def tri_mu(num,points):
    if num == len(points)-1:
        return 1
    return tri_h(num-1,points)/(tri_h(num-1,points)+tri_h(num,points))
def tri_lambda(num,points):
    if num == 0:
        return 1
    return tri_h(num,points)/(tri_h(num-1,points)+tri_h(num,points))
def tri_d(num,points,spec=None):
    if(spec == None):
        return 6*(tri_f(points[num],points[num+1],f_1)-tri_f(points[num-1],points[num],f_1))/(tri_h(num-1,points)+tri_h(num,points))
    else:
        if(num==0):
            return 6*(tri_f(points[0],points[1],f_1)-spec)/tri_h(0,points)
        elif num==len(points)- 1:
            return 6*(spec-tri_f(points[len(points)-2],points[len(points)-1],f_1))/tri_h(len(points)-2,points)
#thomas algorithm
def tri_LU(Martrix,num):
    L=np.zeros((num,num))
    U=np.zeros((num,num))
    for i in range(num):
        U[i][i]=1
        if(i==0):
            L[i][i]=Martrix[i][i]
            U[i][i+1]=Martrix[i][1]/L[i][i]
        else:
            L[i][i-1]=Martrix[i][i-1]#ri
            L[i][i]=Martrix[i][i]-L[i][i-1]*U[i-1][i]#alpha i
            if(i!=num-1):
                U[i][i+1]=Martrix[i][i+1]/L[i][i]#beta
    return L,U


def tri_gen_mart(points,left_df=-1,right_df=1):
    Martrix=np.zeros((len(points),len(points)))
    Vertex=np.zeros((len(points),1))
    Vertex[0][0]=tri_d(0,points,left_df)
    Vertex[len(points)-1][0]=tri_d(len(points)-1,points,right_df)
    for i in range(len(points)):
        Martrix[i][i]=2
        if(i!=len(points)-1):
            Martrix[i][i+1]=tri_lambda(i,points)
            Martrix[i+1][i]=tri_mu(i+1,points)
            if(i!=0):
                Vertex[i][0]=tri_d(i,points)
    L,U=tri_LU(Martrix,len(points))
    y=np.linalg.solve(L,Vertex)
    x=np.linalg.solve(U,y)
    return x
def tri_gen_para(M,p,func=f_1):
    para=np.zeros((len(M)-1,4))
    for i in range(len(M)-1):
        temp_arr=[-p[i+1]]*3
        temp_arr_2=[-p[i]]*3
        for j in range(4):
            param,param2=get_param_langrange(temp_arr,j),get_param_langrange(temp_arr_2,j)
            para[i][j]+=param*(-M[i])/6/tri_h(i,p)
            para[i][j]+=param2*(M[i+1])/6/tri_h(i,p)
        p1=func(p[i])-M[i]*(tri_h(i,p)**2)/6
        p1/=tri_h(i,p)
        p2=func(p[i+1])-M[i+1]*(tri_h(i,p)**2)/6
        p2/=tri_h(i,p)
        para[i][-1]+=p1*p[i+1]-p2*p[i]
        para[i][-2]+=p2-p1
    return para
#
def solve_q1():
    num=11
    p=[i-int((num-1)/2) for i in range(num)]
    x_1=lagrange_interpo(p,f_1)
    contra_plot(x_1,f_1)
def solve_q1_2():
    num = 11
    p = [i - int((num - 1) / 2) for i in range(num)]
    x_1=lagrange_interpo(p,f_1)
    x_2=gen_herm_coeff(x_1,p,df_1,p)
    contra_plot(x_2,f_1,-5.,5.)
def solve_q2():
    x_1=gen_cheby_zeros(11)
    x_2 = lagrange_interpo(x_1, f_1)
    contra_plot(x_2,f_1)
def solve_q3():
    num = 11
    p = [i - int((num - 1) / 2) for i in range(num)]
    para=linear_interpo(p,f_1)
    contra_plot(para,f_1,-5,5,False,p,func_linear)
def solve_q3_2():
    num = 11
    p = [i - int((num - 1) / 2) for i in range(num)]
    para=segmented_hermite_interpo(p)
    contra_plot(para,f_1,-5,5,False,p,func_polygen)
def solve_q4():
    num = 11
    p = [i - int((num - 1) / 2) for i in range(num)]
    M=tri_gen_mart(p,-1,1)
    para=tri_gen_para(M,p)
    contra_plot(para,f_1,-5,5,False,p,func_polygen)
#solve_q1()
#solve_q1_2()
#solve_q2()
#solve_q3()
solve_q4()