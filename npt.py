import numpy as np

ones = np.ones(3)
print(ones)
zeros = np.zeros(3)
print(zeros)
rd = np.random.random(3)
print(rd)

data = np.array([1,2])
ones = np.ones(2)
data_m_noes = data+ones
print(data_m_noes)
print(data.sum())
print(data.min())
print(data.max())

data = np.ones(3)
rdx = np.random.random((3,2))
ts = np.array([
    [0.63457982,0.0383133],
    [0.86345534,0.6418225],
    [0.30245192,0.81960743]
    ]).max(axis=0)


print(ts)