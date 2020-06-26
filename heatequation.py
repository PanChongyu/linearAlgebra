import numpy as np 
import matplotlib.pyplot as plt 

def f(x):
    return -(x-(n_x-1)*dlt_x)*x*20

def g(t):
    return 0

n_x=100
n_t=10000
dlt_x=np.pi/n_x
dlt_t=0.0001
n=10

temp=np.zeros((n_t,n_x))
for t in range(n_t):
    for x_n in range(n_x):
        x=x_n*dlt_x
        temp[t,0]=g(t)
        temp[t,n_x-1]=g(t)
        if (t==0):
            temp[t,x_n]=f(x)
        else:
            if (x_n!=0) and (x_n!=n_x-1):
                temp[t,x_n]=dlt_t*(temp[t-1,x_n-1]+temp[t-1,x_n+1]-2*temp[t-1,x_n])/dlt_x/dlt_x+temp[t-1,x_n]

x=np.zeros(n_x)
for i in range(n_x):
    x[i]=i*dlt_x
y=np.zeros_like(x)
plt.figure()
for i in np.linspace(0,n_t,n):
    k=np.abs(int(i-1))
    print(k)
    plt.plot(x,temp[k])
plt.savefig('heat.png')
