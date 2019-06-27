import numpy as np

n=np.array([10,100])

def jacobi(A,b,n):
    v = np.zeros_like(b)
    for k in range(1000):
        v_new = np.zeros_like(v)
        for i in range(A.shape[0]):
            s1 = A[i, :i].dot(v[:i])
            s2 = A[i, i + 1:].dot(v[i + 1:])
            v_new[i] = (b[i] - s1 - s2) / A[i, i]

        if np.max(np.abs(v - v_new)) < 1e-10:
            break
        v = v_new
    return v

def gs(A, b):
    v = np.zeros_like(b)
    for k in range(1000):
        v_new = np.zeros_like(v)
        for i in range(A.shape[0]):
            s1 = A[i, :i].dot(v[:i])
            s2 = A[i, i + 1:].dot(v[i + 1:])
            v_new[i] = (b[i] - s1 - s2) / A[i, i]

        if np.max(np.abs(v - v_new)) < 1e-10:
            break
        v = v_new
    return v



A = np.array([[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0.0, 3., -1., 8.]])
b = np.array([6., 25., -11., 15.])

v = jacobi(A, b)
print("Solution:", v)
print("Residual:", np.dot(A, v) - b)
