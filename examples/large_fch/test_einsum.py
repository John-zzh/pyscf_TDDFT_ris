import time
import numpy as np
import gc
def use_einsum(eri3c, eri2c):
    start = time.time()
    result = np.einsum('pqP,PQ->pqQ',eri3c, eri2c)
    print(f'use_einsum: {time.time() - start:.2f} seconds')


def normal_dot(eri3c, eri2c):
    start = time.time()
    result = np.dot(eri3c.reshape(occ*vir, nauxbf), eri2c)
    print(f'normal_dot: {time.time() - start:.2f} seconds')

def transpose_dot(eri3c, eri2c):
    start = time.time()
    result3 = np.dot(eri2c, eri3c.reshape(nauxbf, occ*vir))
    print(f'transpose_dot: {time.time() - start:.2f} seconds')


occ = 1600
vir = 1600
nauxbf = 200


eri3c = np.random.rand(occ,vir,nauxbf)
eri2c = np.random.rand(nauxbf,nauxbf)
print('eri3c.flags \n',eri3c.flags)
print('eri2c.flags \n',eri2c.flags)

# use_einsum(eri3c, eri2c)
normal_dot(eri3c, eri2c)

gc.collect()

eri3c = np.random.rand(nauxbf, occ, vir)
eri2c = np.random.rand(nauxbf,nauxbf)
transpose_dot(eri3c, eri2c)