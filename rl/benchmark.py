from timeit import default_timer as timer
import numpy.random as npr
import numpy as np
from tqdm import trange
def foo():
    N= 10000
    M = [npr.randint(255, size=(84,84)) for _ in range(N)]
    start = timer()
    batch_size = 32
    A = np.zeros((batch_size, 84, 84))
    for _ in trange(50000000):
        L = npr.choice(N, batch_size)
        #e = np.stack([M[i] for i in L])
        for j, i in enumerate(L):
            A[j]=M[i]
        #M0 = [np.stack(M[i:i+64]) for i in range(10000//32)]
        #print(M0[0].shape)
    time = timer() - start
    print(time)

def foo1():
    N= 10000
    M = npr.randint(N, size=(N, 84,84)) 
    start = timer()
    batch_size = 32
    A = np.zeros((batch_size, 84, 84))
    for _ in trange(50000000):
        L = npr.choice(N, batch_size)
        #e = np.stack(M[np.array(L)])
        e = M[np.array(L)]
        print(e.shape)
    time = timer() - start
    print(time)
foo1()
