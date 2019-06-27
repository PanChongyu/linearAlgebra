import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt 

p_array=np.array([4])
a=1
omega=2/3

def eigenvector_coeff(u):
    """Coordinates of u in the basis of eigenvectors of the Jacobi iteration matrix:
       w_{m,i} = sin(m i pi / N)
       Assumes u[0] == u[N] == 0."""

    N = len(u) - 1            # u stores u_0, ..., u_N
    a = np.zeros_like(u)      # allocated storage
    i = np.arange(0, len(u))
    for m in range(0, N):
        a[m] = np.dot(u, np.sin(m * i * np.pi / N)) * 2 / N

    return a

# `eigenvector_coeff` can be more efficiently computed using the Fast Fourier Tranform.
# This takes O(N log(N)) time, while eigenvector_coeff takes O(N^2) time.
def sin_fft(u):
    """Sine FFT of u. Assumes u[0] == u[-1] == 0."""
    return -np.imag(np.fft.fft(np.concatenate((u[:-1], -u[:0:-1]))))[:len(u) - 1] * (1. / (len(u) - 1))

# Make a nice plot of the spectrum (coefficients in the eigenvector basis)
def plot_spectrum(data,na, labels = None):
    plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0.))
    ax.spines['right'].set_color('none')
    ax.set_xlabel('$m$')
    ax.set_ylabel('$a_m$')

    for i, d in enumerate(data):
        if labels:
            label = labels[i]
        else:
            label = None
        ax.plot(eigenvector_coeff(d), 'o', label=label)

    if labels:
        ax.legend()
    plt.savefig(na)

def Amul(v,n):
    aii=2+a/n/n
    return np.pad(aii*v[1:-1]-v[:-2]-v[2:],1,'constant')

def f(x):
    return (np.sin(12*np.pi*x)+np.sin(2*np.pi*x)+np.sin(np.pi*x))*(x**8)

def dampedJacobi(v_temp,b,n):
    v=v_temp.copy()
    aii=2*n**2+a
    v[1:-1]=(1-omega)*v[1:-1]+omega*(b[1:-1]+n**2*(v[:-2]+v[2:]))/aii
    return v

def GS(v_temp,b,n):
    v=v_temp.copy()
    aii=2+a/n/n
    for i in range(n-1):
        v[i+1]=(v[i]+v[i+2]+b[i+1]/n/n)/aii
    return v

def restrict(v):
    return np.pad(0.25*(v[1:-3:2]+v[3:-1:2])+0.5*v[2:-2:2],1,'constant')

def prolong(v):
    v_new=np.zeros(2*len(v)-1)
    v_new[::2]=v
    v_new[1::2]=0.5*(v[:-1]+v[1:])
    return v_new

def vcycle_j(v_temp,b,p):
    v=v_temp.copy()        
    n=2**p
    for i in range(4):
        v=dampedJacobi(v,b,n)
    if (p==1):
        return v
    r=b-Amul(v,n)*n*n
    r2=restrict(r)
    e=np.zeros_like(r2)
    v2=vcycle_j(e,r2,p-1)
    v=v+prolong(v2)
    for i in range(4):
        v=dampedJacobi(v,b,n)
    return v

def vcycle_g(v_temp,b,p):
    v=v_temp.copy()        
    n=2**p
    for i in range(4):
        v=GS(v,b,n)
    if (p==1):
        return v
    r=b-Amul(v,n)*n*n
    r2=restrict(r)
    e=np.zeros_like(r2)
    v2=vcycle_g(e,r2,p-1)
    v=v+prolong(v2)
    for i in range(4):
        v=GS(v,b,n)
    return v

def para(p):
    n=2**p
    b=np.zeros((n+1))
    for i in range(n+1):
        if (i==0) or (i==n):
            b[i]=0
        else:
            b[i]=f(i/n)
    return n,b

def norm(v):
    return np.sqrt(np.sum(v**2))

for p in p_array:
    n,b=para(p)
    v=np.zeros((n+1))
    v1=vcycle_j(v,b,p)
    v2=vcycle_j(v1,b,p)
    r=b/n/n-Amul(v,n)
    r1=b/n/n-Amul(v1,n)
    r2=b/n/n-Amul(v2,n)
    print('dampJacobi',norm(r1)/norm(r2),norm(r)/norm(r1))
    print(norm(r1*n*n),norm(r2*n*n))

    v=np.zeros((n+1))
    v1=vcycle_g(v,b,p)
    v2=vcycle_g(v1,b,p)
    rr1=b/n/n-Amul(v,n)
    r11=b/n/n-Amul(v1,n)
    r12=b/n/n-Amul(v2,n)
    print('GS',norm(r11)/norm(r12),norm(rr1)/norm(r11))

plot_spectrum([r1*n*n,r2*n*n],'damp.png')
plot_spectrum([r11*n*n,r12*n*n],'GS.png')