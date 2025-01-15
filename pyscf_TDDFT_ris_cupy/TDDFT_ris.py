from pyscf import gto, lib, dft, df
import scipy
import numpy as np
import cupy as cp
import multiprocessing as mp
from pyscf_TDDFT_ris_cupy import parameter, math_helper, spectralib, eigen_solver

# from pyscf_TDDFT_ris import eigen_solver_old as eigen_solver
# from pyscf_TDDFT_ris import eigen_solver

import time

cp.set_printoptions(linewidth=250, threshold=cp.inf)

einsum = cp.einsum

num_cores = int(mp.cpu_count())
print("This job can use: " + str(num_cores) + "CPUs")


# def print_memory_usage(line):
#     import psutil
#     process = psutil.Process()
#     memory_usage = process.memory_info().rss  
#     print(f"memory usage at {line}: {memory_usage / (1024 ** 2):.0f} MB")

def get_auxmol(mol, theta=0.2, fitting_basis='s'):
    """
    Assigns a minimal auxiliary basis set to the molecule.

    Args:
        mol: The input molecule object.
        theta: The scaling factor for the exponents.
        fitting_basis: Basis set type ('s', 'sp', 'spd').

    Returns:
        auxmol: The molecule object with assigned auxiliary basis.
    """
    print(f'Asigning minimal auxiliary basis set: {fitting_basis}')
    print(f'The exponent alpha: {theta}/R^2 ')

    
    '''
    parse_arg = False 
    turns off PySCF built-in parsing function
    '''
    tt = time.time()
    auxmol = gto.M(atom=mol.atom, 
                    basis=mol.basis,
                    parse_arg=False, 
                    spin=mol.spin, 
                    charge=mol.charge,
                    cart=mol.cart)
    # print(f'auxmol build time = {time.time()-tt:.1f} seconds')
    auxmol_basis_keys = mol._basis.keys()

    '''
    auxmol_basis_keys: (['C1', 'H2', 'O3', 'H4', 'H5', 'H6'])
    
    aux_basis:
    C1 [[0, [0.1320292535005648, 1.0]]]
    H2 [[0, [0.1999828038466018, 1.0]]]
    O3 [[0, [0.2587932305664396, 1.0]]]
    H4 [[0, [0.1999828038466018, 1.0]]]
    H5 [[0, [0.1999828038466018, 1.0]]]
    H6 [[0, [0.1999828038466018, 1.0]]]
    '''
    aux_basis = {}
    for atom_index in auxmol_basis_keys:
        atom = ''.join([char for char in atom_index if char.isalpha()])
        '''
        exponent_alpha = theta/R^2 
        '''
        exp_alpha = parameter.ris_exp[atom] * theta

        if 's' in fitting_basis:
            aux_basis[atom_index] = [[0, [exp_alpha, 1.0]]]

        if atom != 'H':
            if 'p' in fitting_basis:
                aux_basis[atom_index].append([1, [exp_alpha, 1.0]])
            if 'd' in fitting_basis:
                aux_basis[atom_index].append([2, [exp_alpha, 1.0]])

    auxmol.basis = aux_basis
    auxmol.build()
    print('   auxmol.cart =', mol.cart)
    # print('=====================')
    # [print(k, v) for k, v in auxmol._basis.items()]
    return auxmol


def calculate_batches_by_basis_count(mol, max_nbf_per_batch):
    """
    根据基函数数目划分 batch，并映射回最接近的 shell 数目。
    
    参数:
        mol: PySCF 的 Mole 对象。
        max_nbf_per_batch: 每个 batch 的最大基函数数目。
    
    返回:
        一个列表，其中每个元素是 (start_shell, end_shell) 的元组，表示每个 batch 的 shell 范围。
    """
    ao_loc = mol.ao_loc  # Shell 到基函数的映射
    nbas = mol.nbas      # Shell 数量
    
    batches = []
    current_batch_start = 0
    current_nbf = 0  # 当前 batch 的累计基函数数目
    
    for i in range(nbas):
        # 当前 shell 的基函数数目
        nbf_in_shell = ao_loc[i+1] - ao_loc[i]
        if current_nbf + nbf_in_shell > max_nbf_per_batch:
            # 如果当前 batch 超出基函数限制，结束当前 batch
            batches.append((current_batch_start, i))
            current_batch_start = i
            current_nbf = 0
        
        # 累计当前 shell 的基函数数目
        current_nbf += nbf_in_shell
    
    # 处理最后一个 batch
    if current_batch_start < nbas:
        batches.append((current_batch_start, nbas))
    
    return batches


def get_eri2c_eri3c(mol, auxmol, max_mem_mb, omega=0, single=True):

    '''
    (uv|kl) = Σ_PQ (uv|P)(P|Q)^-1(Q|kl)
    2 center 2 electron AO integral (P|Q)
    nauxbf * nauxbf

    3 center 2 electron integral (uv|P)
    (nbf, nbf, nauxbf)

    actually it returns (P|uv) ( or (Q|kl), they are the same, just transposed)
    (nauxbf, nbf, nbf)

    '''

    start = time.time()
    nbf = mol.nao_nr()
    nauxbf = auxmol.nao_nr()
    dtype = cp.float32 if single else cp.float64


    if omega != 0:
        mol.set_range_coulomb(omega)
        auxmol.set_range_coulomb(omega)

    tag = '_cart' if mol.cart else '_sph'

    ''' eri2c is samll enough, just do it incore''' 
    eri2c = auxmol.intor('int2c2e' + tag)
    eri2c = cp.asarray(eri2c, dtype=cp.float64, order='C')

    pmol = mol + auxmol
    pmol.cart = mol.cart

    '''for eri3c, if it is too large, then we need to compute in batches and return a generator '''
    full_eri3c_mem = nbf * nbf * nauxbf * 8 / (1024 ** 2)
    print(f'    Full eri3c in shape {int(nbf), int(nbf), int(nauxbf)}, will take {full_eri3c_mem:.0f} MB')

    max_mem_for_one_batch = max_mem_mb / 2
    print('    max_mem_for_one_batch', max_mem_for_one_batch)
    # print('    auxmol.nbas, stride', auxmol.nbas, stride)
    if max_mem_for_one_batch >= full_eri3c_mem:
        print('    small 3c2e, just incore')
        tt = time.time()
        shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + auxmol.nbas)
        eri3c = pmol.intor('int3c2e' + tag, shls_slice=shls_slice)
        print(f'    eri3c on CPU time: {time.time() - tt:.2f} s')
        tt = time.time()

        eri3c = eri3c.transpose(2, 1, 0)
        eri3c = eri3c.astype(dtype=dtype, order='C')
        # eri3c = cp.asarray(eri3c)
        print(f'    eri3c to GPU time: {time.time() - tt:.2f} s')

        return eri2c, eri3c

    else:
        max_nbf_per_batch = int(max_mem_for_one_batch / (nbf * nbf * 8 / (1024 ** 2)))
        print('    max_nbf_per_batch', max_nbf_per_batch)
        batches = calculate_batches_by_basis_count(auxmol, max_nbf_per_batch)

        print(f'    large 3c2e, batch generator, in {len(batches)} batches')
        def eri3c_batch_generator():
            # for start_aux in range(0, auxmol.nbas, stride):
            #     end_aux = min(start_aux + stride, auxmol.nbas)
            for start_shell, end_shell in batches:
                shls_slice = (
                    0, mol.nbas,            # First dimension (mol)
                    0, mol.nbas,            # Second dimension (mol)
                    mol.nbas + start_shell,   # Start of aux basis
                    mol.nbas + end_shell      # End of aux basis (exclusive)
                )
                tt = time.time()
                eri3c_slice = pmol.intor('int3c2e' + tag, shls_slice=shls_slice)
                eri3c_slice = eri3c_slice.transpose(2, 1, 0)
                eri3c_slice = eri3c_slice.astype(dtype=dtype, order='C')
                print(f'    eri3c_slice on CPU time: {time.time() - tt:.2f} s')
                tt = time.time()
                # eri3c_slice = cp.asarray(eri3c_slice)
                print(f'    eri3c_slice to GPU time: {time.time() - tt:.2f} s')
                print(f'    eri3c_slice mem: {eri3c_slice.nbytes / (1024 ** 2):.0f} MB')

                yield eri3c_slice
        return eri2c, eri3c_batch_generator

   
def get_eri2c_eri3c_RSH(mol, auxmol, eri2c_K, eri3c_K, alpha, beta, omega, max_mem_mb, single=False):

    '''
    in the RSH functional, the Exchange ERI splits into two parts
    (ij|ab) = (ij|1-(alpha + beta*erf(omega))/r|ab) + (ij|alpha + beta*erf(omega)/r|ab)
    -- The first part (ij|1-(alpha + beta*erf(omega))/r|ab) is short range, 
        treated by the DFT XC functional, thus not considered here
    -- The second part is long range 
        (ij|alpha + beta*erf(omega)/r|ab) = alpha (ij|r|ab) + beta*(ij|erf(omega)/r|ab)


        eri2c_K, eri3c_K are omega = 0, just like usual eri2c, eri3c
    '''            
    print('    generating 2c2e_RSH and 3c2e_RSH for RI-K (ij|ab) ...')
    eri2c_erf, eri3c_erf = get_eri2c_eri3c(mol=mol, auxmol=auxmol, max_mem_mb=max_mem_mb, omega=omega, single=single)
    eri2c_RSH = alpha * eri2c_K + beta * eri2c_erf
    # print('type(eri3c_K)', type(eri3c_K))
    # print('type(eri3c_erf)', type(eri3c_erf))
    # print('callable(eri3c_K) and callable(eri3c_erf)', callable(eri3c_K),  callable(eri3c_erf))

    if isinstance(eri3c_K, cp.ndarray) and isinstance(eri3c_erf, cp.ndarray):
        eri3c_RSH = alpha * eri3c_K + beta * eri3c_erf
        return eri2c_RSH, eri3c_RSH

    elif callable(eri3c_K) and callable(eri3c_erf): 
        def eri3c_RSH_generator():
            eri3c_K_gen = eri3c_K()
            eri3c_erf_gen = eri3c_erf()
            while True:
                try:
                    eri3c_K_batch = next(eri3c_K_gen)
                    eri3c_erf_batch = next(eri3c_erf_gen)
                    eri3c_RSH_batch = alpha * eri3c_K_batch + beta * eri3c_erf_batch
                    yield eri3c_RSH_batch
                except StopIteration:
                    break
        return eri2c_RSH, eri3c_RSH_generator
    else:
        raise ValueError('eri3c_K and eri3c_erf must be both cp.ndarray or callable')

    

'''
            n_occ          n_vir
       -|-------------||-------------|
        |             ||             |
  n_occ |   3c2e_ij   ||  3c2e_ia    |
        |             ||             |
        |             ||             |
       =|=============||=============|
        |             ||             |
  n_vir |             ||  3c2e_ab    |
        |             ||             |
        |             ||             |
       -|-------------||-------------|
'''

def get_pre_Tpq_one_batch(eri3c, C_p, C_q):
    '''    
    T_pq means rank-3 Tensor (pq|P)

    T_pq = Σ_uvQ C_u^p (uv|Q)L_PQ  C_v^q

    preT_pq^P =  Σ_uv C_up (uv|Q) C_vq, thus is the return 

    C_p and C_q:  C[:, :n_occ] or C[:, n_occ:], can be both

    lower_inv_eri2c (L_PQ): (P|Q)^-1 = LL^T, not contract in this function

    The following code is doing this:
    eri3c_Cp = einsum("Puv, up -> Ppv", eri3c, C_p)
    pre_T_pq = einsum("Ppv,vq->Ppq", eri3c_Cp, C_q)

    T_pq = einsum("PQ,Ppq->Qpq", lower_inv_eri2c, pre_T_pq)

    T_pq finally reshape to pq P shape

    it has many manually reshape and transpose 
    beause respecting c-contiguous is beneficial for memory access, much faster

    '''
    C_p = C_p.get()
    C_q = C_q.get()

    print('type(eri3c), type(C_p), type(C_q)', type(eri3c), type(C_p), type(C_q))
    t_satrt = time.time()

    tt = time.time()
    '''eri3c in shape (nauxbf, nbf, nbf)'''
    nbf = eri3c.shape[1]
    nauxbf = eri3c.shape[0]

    n_p = C_p.shape[1]
    n_q = C_q.shape[1]


    '''eri3c (nauxbf, nbf, nbf) -> (nauxbf*nbf, nbf)
       C_p (nbf, n_p)
       >> eri3c_C_p (nauxbf*nbf, n_p)'''
    eri3c = eri3c.reshape(nauxbf*nbf, nbf)
    eri3c_C_p = np.dot(eri3c, C_p)
    tt = time.time()

    ''' eri3c_C_p (nauxbf*nbf, n_p) 
        -> (nauxbf, nbf, n_p) 
        -> (nauxbf, n_p, nbf) '''
    eri3c_C_p = eri3c_C_p.reshape(nauxbf, nbf, n_p)
    eri3c_C_p = eri3c_C_p.transpose(0,2,1)

    ''' eri3c_C_p  (nauxbf, n_p, nbf) -> (nauxbf*n_p, nbf)
        C_q  (nbf, n_q)
        >> pre_T_pq (nauxbf*n_p, n_q) >  (nauxbf, n_p, n_q)  '''
    eri3c_C_p = eri3c_C_p.reshape(nauxbf*n_p, nbf)
    pre_T_pq = np.dot(eri3c_C_p, C_q)
    pre_T_pq = pre_T_pq.reshape(nauxbf, n_p, n_q)
    pre_T_pq = cp.asarray(pre_T_pq)
    print(f'    pre_T_pq time: {time.time() - t_satrt:.1f} seconds')
    return pre_T_pq

def get_pre_T_pq_to_Tpq(pre_T_pq: cp.ndarray, lower_inv_eri2c: cp.ndarray):
    ''' pre_T_pq  (nauxbf, n_p, n_q) -> (nauxbf, n_p*n_q)
        lower_inv_eri2c  (nauxbf, nauxbf)
        >> T_pq (nauxbf, n_p*n_q) -> (nauxbf, n_p, n_q)'''
    tt = time.time()
    nauxbf, n_p, n_q = pre_T_pq.shape

    pre_T_pq = pre_T_pq.reshape(nauxbf, n_p*n_q)

    T_pq = cp.dot(lower_inv_eri2c.T, pre_T_pq)
    T_pq = T_pq.reshape(nauxbf, n_p, n_q)

    print(f'pre_T_pq_to_Tpq one batch time {time.time() - tt:.1f} seconds')
    return T_pq 

def get_Tpq(eri3c, lower_inv_eri2c, C_p, C_q):
    """
    Wrapper function to handle eri3c as either a matrix or a callable.
    """
    # 判断 eri3c 的类型
    # print('type(eri3c)', type(eri3c))
    if not callable(eri3c):
        pre_T_pq = get_pre_Tpq_one_batch(eri3c, C_p, C_q)


    elif callable(eri3c):
        # 分批次计算 T_pq
        nauxbf = lower_inv_eri2c.shape[0]
        n_p = C_p.shape[1]
        n_q = C_q.shape[1]

        pre_T_pq = np.zeros((nauxbf, n_p, n_q), dtype=C_p.dtype) 
        aux_offset = 0  # Offset to track where to store results in T_pq
        
        i = 0
        for eri3c_batch in eri3c():
            print(f'        batch {i} done')
            i += 1
            # Process each batch of eri3c
            batch_size = eri3c_batch.shape[0]
            pre_T_pq[aux_offset:aux_offset + batch_size, :, :] = get_pre_Tpq_one_batch(
                eri3c_batch, C_p, C_q
            )
            aux_offset += batch_size  # Update the offset for the next batch

    else:
        raise ValueError("eri3c must be either a numpy.ndarray or a callable returning a generator.")

    pre_T_pq = cp.asarray(pre_T_pq)
    T_pq = get_pre_T_pq_to_Tpq(pre_T_pq, lower_inv_eri2c)
    return T_pq


def gen_hdiag_MVP(hdiag, n_occ, n_vir):
    # def hdiag_MVP(V):
    #     delta_hdiag_v = einsum("ia,iam->iam", delta_hdiag, V)
    #     return delta_hdiag_v
    def hdiag_MVP(V):
        m = V.shape[0]
        V = V.reshape(m, n_occ*n_vir)
        hdiag_v = hdiag * V
        hdiag_v = hdiag_v.reshape(m, n_occ, n_vir)
        return hdiag_v
    
    return hdiag_MVP
    

def gen_iajb_MVP(T_left, T_right):
    def iajb_MVP(V):
        '''
        (ia|jb) = Σ_Pjb (T_left_ia^P T_right_jb^P V_jb^m)
                = Σ_P [ T_left_ia^P Σ_jb(T_right_jb^P V_jb^m) ]
        if T_left == T_right, then it is either 
            (1) (ia|jb) in RKS 
            or 
            (2)(ia_α|jb_α) or (ia_β|jb_β) in UKS, 
        elif T_left != T_right
            it is (ia_α|jb_β) or (ia_β|jb_α) in UKS

        V in shape (m, n_occ * n_vir)
        '''
        
        T_right_jb_V = einsum("Pjb,mjb->Pm", T_right, V)
        iajb_V = einsum("Pia,Pm->mia", T_left, T_right_jb_V)

        return iajb_V
    return iajb_MVP

def gen_ijab_MVP(T_ij, T_ab):
    def ijab_MVP(V):
        '''
        (ij|ab) = Σ_Pjb (T_ij^P T_ab^P V_jb^m)
                = Σ_P [T_ij^P Σ_jb(T_ab^P V_jb^m)]
        V in shape (m, n_occ * n_vir)
        '''

        T_ab_V = einsum("Pab,mjb->Pamj", T_ab, V)
        ijab_V = einsum("Pij,Pamj->mia", T_ij, T_ab_V)

        return ijab_V
    return ijab_MVP

def get_ibja_MVP(T_ia):    
    def ibja_MVP(V):
        '''
        the exchange (ib|ja) in B matrix
        (ib|ja) = Σ_Pjb (T_ib^P T_ja^P V_jb^m)
                = Σ_P [T_ja^P Σ_jb(T_ib^P V_jb^m)]           
        '''

        T_ib_V = einsum("Pib,mjb->Pimj", T_ia, V)
        ibja_V = einsum("Pja,Pimj->mia", T_ia, T_ib_V)

        return ibja_V
    return ibja_MVP


class TDDFT_ris(object):
    def __init__(self, 
                mf, 
                theta: float = 0.2,
                J_fit: str = 's',
                K_fit: str = 's',
                Ktrunc: float = 0,
                a_x: float = None,
                omega: float = None,
                alpha: float = None,
                beta: float = None,
                conv_tol: float = 1e-5,
                nroots: int = 5,
                max_iter: int = 25,
                spectra: bool = True,
                pyscf_TDDFT_vind: callable = None,
                out_name: str = '',
                print_threshold: float = 0.05,
                GS: bool = False,
                single: bool = False,
                max_mem_mb: int = 8000,):

        self.mf = mf
        print('type mf', type(mf))
        print('type mf.mol', type(mf.mol))
        self.theta = theta
        self.J_fit = J_fit
        self.K_fit = K_fit

        if J_fit == K_fit: 
            print(f'use same J and K fitting basis: {J_fit}')
        else: 
            print(f'use different J and K fitting basis: J with {J_fit} and K with {K_fit}')

        self.Ktrunc = Ktrunc
        self.a_x = a_x
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.conv_tol = conv_tol
        self.nroots = nroots
        self.max_iter = max_iter
        self.mol = mf.mol
        self.pyscf_TDDFT_vind = pyscf_TDDFT_vind
        self.spectra = spectra
        self.out_name = out_name
        self.print_threshold = print_threshold
        self.GS = GS
        self.single = single
        self.max_mem_mb = max_mem_mb
        print('self.nroots', self.nroots)
        print('use single precision?', self.single)


        if hasattr(mf, 'xc'):
            functional = mf.xc.lower()
            self.functional = mf.xc
            print('loading default XC functional paramters from parameter.py')
            if functional in parameter.rsh_func.keys():
                '''
                RSH functional, need omega, alpha, beta
                '''
                print('use range-separated hybrid XC functional')
                omega, alpha, beta = parameter.rsh_func[functional]
                self.a_x = 1
                self.omega = omega
                self.alpha = alpha
                self.beta = beta

            elif functional in parameter.hbd_func.keys():
                print('use hybrid XC functional')
                self.a_x = parameter.hbd_func[functional]

            else:
                raise ValueError(f"I do not have paramters for XC functional {mf.xc} yet, please either manually input HF component a_x or add parameters in the parameter.py file")
            
        else:
            if self.a_x == None and self.omega == None and self.alpha == None and self.beta == None:
                raise ValueError('Please specify the XC functional name or the XC functional parameters')
            else:
                if a_x:
                    self.a_x = a_x
                    print("hybrid XC functional")
                    print(f"manually input HF component ax = {a_x}")

                elif omega and alpha and beta:
                    self.a_x = 1
                    self.omega = omega
                    self.alpha = alpha
                    self.beta = beta
                    print("range-separated hybrid XC functional")
                    print(f"manually input ω = {self.omega}, screening factor")
                    print(f"manually input α = {self.alpha}, fixed HF exchange contribution")
                    print(f"manually input β = {self.beta}, variable part")

                else:
                    raise ValueError('missing parameters for range-separated XC functional, please input (w, al, be)')
        print('self.a_x', self.a_x)
        if self.mol.cart:
            self.eri_tag = '_cart'
        else:
            self.eri_tag = '_sph'
        print('cartesian or spherical electron integral =',self.eri_tag)


        if single == True:
            mf.mo_coeff = mf.mo_coeff.astype(cp.float32)

        if mf.mo_coeff.ndim == 2:
            self.RKS = True
            self.UKS = False
            # self.n_if = len(mf.mo_occ)
            n_occ = int(sum(mf.mo_occ>0))
            n_vir = int(sum(mf.mo_occ==0))
            print('type(n_occ)', type(n_occ))    
            print('type(n_vir)', type(n_vir))
            self.n_occ = n_occ
            self.n_vir = n_vir

            self.C_occ_notrunc = cp.asfortranarray(mf.mo_coeff[:,:n_occ])
            self.C_vir_notrunc = cp.asfortranarray(mf.mo_coeff[:,n_occ:])
            mo_energy = mf.mo_energy
            print('type(mo_energy)', type(mo_energy))
            print('mo_energy.shape', mo_energy.shape)
            vir_ene = mo_energy[n_occ:].reshape(1,n_vir)
            occ_ene = mo_energy[:n_occ].reshape(n_occ,1)
            delta_hdiag = cp.repeat(vir_ene, n_occ, axis=0) - cp.repeat(occ_ene, n_vir, axis=1)
            self.delta_hdiag = delta_hdiag

            print('n_occ =', n_occ)
            print('n_vir =', n_vir)

            if Ktrunc > 0:
                print(f' MO truncation in K with threshold {Ktrunc} eV above HOMO and below LUMO')

                trunc_tol_au = Ktrunc/parameter.Hartree_to_eV     

                homo_vir_delta_ene = delta_hdiag[-1,:]
                occ_lumo_delta_ene = delta_hdiag[:,0]
                
                rest_occ = cp.sum(occ_lumo_delta_ene <= trunc_tol_au)
                rest_vir = cp.sum(homo_vir_delta_ene <= trunc_tol_au)
                
            elif Ktrunc == 0: 
                print('no MO truncation in K')
                rest_occ = n_occ
                rest_vir = n_vir 
                
            print('rest_occ =',rest_occ)
            print('rest_vir =',rest_vir)

            self.C_occ_Ktrunc = cp.asfortranarray(mf.mo_coeff[:,n_occ-rest_occ:n_occ])
            self.C_vir_Ktrunc = cp.asfortranarray(mf.mo_coeff[:,n_occ:n_occ+rest_vir])
             
            self.rest_occ = rest_occ
            self.rest_vir = rest_vir

        elif mf.mo_coeff.ndim == 3:
            self.RKS = False
            self.UKS = True
            # self.n_if = len(mf.mo_occ[0])
            self.n_occ_a = sum(mf.mo_occ[0]>0)
            self.n_vir_a = sum(mf.mo_occ[0]==0)
            self.n_occ_b = sum(mf.mo_occ[1]>0)
            self.n_vir_b = sum(mf.mo_occ[1]==0)
            print('n_occ for alpha spin =',self.n_occ_a)
            print('n_vir for alpha spin =',self.n_vir_a)
            print('n_occ for beta spin =',self.n_occ_b)
            print('n_vir for beta spin =',self.n_vir_b)    



    #  ===========  RKS hybrid ===========
    def get_RKS_TDA_hybrid_MVP(self):
        ''' TDA RKS hybrid '''
        a_x = self.a_x
        n_occ = self.n_occ
        n_vir = self.n_vir

        mo_coeff = self.mf.mo_coeff
        mo_energy = self.mf.mo_energy

        single = self.single 

        C_occ_notrunc = self.C_occ_notrunc
        C_vir_notrunc = self.C_vir_notrunc

        C_occ_Ktrunc = self.C_occ_Ktrunc
        C_vir_Ktrunc  = self.C_vir_Ktrunc

        rest_occ = self.rest_occ
        rest_vir = self.rest_vir

        hdiag = cp.asarray(self.delta_hdiag.reshape(-1))

        mol = self.mol
        theta = self.theta 

        J_fit = self.J_fit
        K_fit = self.K_fit

        alpha = self.alpha
        beta = self.beta
        omega = self.omega

        max_mem_mb = self.max_mem_mb
        ''' RIJ '''
        tt = time.time()
        print('==================== RIJ ====================')
        auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)
        
        eri2c_J, eri3c_J = get_eri2c_eri3c(mol=mol, auxmol=auxmol_J, omega=0, single=single, max_mem_mb=max_mem_mb)

        lower_inv_eri2c_J = math_helper.matrix_power(eri2c_J,-0.5,epsilon=1e-6)
        lower_inv_eri2c_J = lower_inv_eri2c_J.astype(dtype=cp.float32 if single else cp.float64)
       
        T_ia_J = get_Tpq(eri3c=eri3c_J, lower_inv_eri2c=lower_inv_eri2c_J, C_p=C_occ_notrunc, C_q=C_vir_notrunc)
        T_ia_J = cp.asarray(T_ia_J)
        print(f'T_ia_J time {time.time() - tt:.1f} seconds')
        tt = time.time()

        print('==================== RIK ====================')
        ''' RIK '''
        if K_fit == J_fit:
            ''' K uese exactly same basis as J and they share same set of Tensors'''
            auxmol_K = auxmol_J
            eri2c_K, eri3c_K = eri2c_J, eri3c_J
        else:
            ''' K uese different basis as J'''
            auxmol_K = get_auxmol(mol=mol, theta=theta, fitting_basis=K_fit) 
            eri2c_K, eri3c_K = get_eri2c_eri3c(mol=mol, auxmol=auxmol_K, omega=0, single=single, max_mem_mb=max_mem_mb)

        if omega and omega > 0:
            print(f'        rebuild eri2c_K and eri3c_K with screening factor ω = {omega}')
            '''RSH, eri2c_K and eri3c_K need to be redefined'''
            eri2c_K, eri3c_K = get_eri2c_eri3c_RSH(mol=mol, 
                                                auxmol=auxmol_K, 
                                                eri2c_K=eri2c_K, 
                                                eri3c_K=eri3c_K, 
                                                alpha=alpha, 
                                                beta=beta, 
                                                omega=omega, 
                                                single=single,
                                                max_mem_mb=max_mem_mb)

        lower_inv_eri2c_K = math_helper.matrix_power(eri2c_K,-0.5,epsilon=1e-6)
        lower_inv_eri2c_K = lower_inv_eri2c_K.astype(dtype=cp.float32 if single else cp.float64)


        T_ij_K = get_Tpq(eri3c=eri3c_K, lower_inv_eri2c=lower_inv_eri2c_K, C_p=C_occ_Ktrunc, C_q=C_occ_Ktrunc)
        T_ab_K = get_Tpq(eri3c=eri3c_K, lower_inv_eri2c=lower_inv_eri2c_K, C_p=C_vir_Ktrunc, C_q=C_vir_Ktrunc)

        T_ij_K = cp.asarray(T_ij_K)
        T_ab_K = cp.asarray(T_ab_K)
        print(f'T_ij_K T_ab_K time {time.time() - tt:.1f} seconds')

        # hdiag = delta_hdiag.reshape(-1)
        hdiag_MVP = gen_hdiag_MVP(hdiag=hdiag, n_occ=n_occ, n_vir=n_vir)

        iajb_MVP = gen_iajb_MVP(T_left=T_ia_J, T_right=T_ia_J)
        ijab_MVP = gen_ijab_MVP(T_ij=T_ij_K,   T_ab=T_ab_K)

        def RKS_TDA_hybrid_MVP(X):
            ''' hybrid or range-sparated hybrid, a_x > 0
                return AX
                AV = hdiag_MVP(V) + 2*iajb_MVP(V) - a_x*ijab_MVP(V)
                for RSH, a_x = 1

                if not MO truncation, then n_occ-rest_occ=0 and rest_vir=n_vir
            '''
            # print('a_x=', a_x)
            nstates = X.shape[0]
            X = X.reshape(nstates, n_occ, n_vir)
            AX = hdiag_MVP(X) 
            AX += 2 * iajb_MVP(X) 

            AX[:,n_occ-rest_occ:,:rest_vir] -= a_x * ijab_MVP(X[:,n_occ-rest_occ:,:rest_vir])
            AX = AX.reshape(nstates, n_occ*n_vir)

            return AX

        return RKS_TDA_hybrid_MVP, hdiag
            

    def gen_RKS_TDDFT_hybrid_MVP(self):  
        '''hybrid RKS TDDFT'''
        a_x = self.a_x
        n_occ = self.n_occ
        n_vir = self.n_vir

        mo_coeff = self.mf.mo_coeff
        mo_energy = self.mf.mo_energy

        single = self.single 

        C_occ_notrunc = self.C_occ_notrunc
        C_vir_notrunc = self.C_vir_notrunc

        C_occ_Ktrunc = self.C_occ_Ktrunc
        C_vir_Ktrunc  = self.C_vir_Ktrunc

        rest_occ = self.rest_occ
        rest_vir = self.rest_vir

        hdiag = cp.asarray(self.delta_hdiag.reshape(-1))

        mol = self.mol
        theta = self.theta 

        J_fit = self.J_fit
        K_fit = self.K_fit

        alpha = self.alpha
        beta = self.beta
        omega = self.omega

        max_mem_mb = self.max_mem_mb
        ''' RIJ '''
        tt = time.time()
        print('==================== RIJ ====================')
        auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)
        
        eri2c_J, eri3c_J = get_eri2c_eri3c(mol=mol, auxmol=auxmol_J, omega=0, single=single, max_mem_mb=max_mem_mb)

        lower_inv_eri2c_J = math_helper.matrix_power(eri2c_J,-0.5,epsilon=1e-6)
        lower_inv_eri2c_J = lower_inv_eri2c_J.astype(dtype=cp.float32 if single else cp.float64)
       
        unit = 4 if single else 8
        print(f'T_ia_J is going to take { auxmol_J.nao_nr() * n_occ * n_vir * unit / (1024 ** 2):.0f} MB memory')
        T_ia_J = get_Tpq(eri3c=eri3c_J, lower_inv_eri2c=lower_inv_eri2c_J, C_p=C_occ_notrunc, C_q=C_vir_notrunc)

        T_ia_J = cp.array(T_ia_J)
        print(f'T_ia_J time {time.time() - tt:.1f} seconds')
        tt = time.time()

        print('==================== RIK ====================')
        ''' RIK '''
        if K_fit == J_fit:
            ''' K uese exactly same basis as J and they share same set of Tensors'''
            auxmol_K = auxmol_J
            eri2c_K, eri3c_K = eri2c_J, eri3c_J
        else:
            ''' K uese different basis as J'''
            auxmol_K = get_auxmol(mol=mol, theta=theta, fitting_basis=K_fit) 
            eri2c_K, eri3c_K = get_eri2c_eri3c(mol=mol, auxmol=auxmol_K, omega=0, single=single, max_mem_mb=max_mem_mb)

        if omega and omega > 0:
            print(f'        rebuild eri2c_K and eri3c_K with screening factor ω = {omega}')
            '''RSH, eri2c_K and eri3c_K need to be redefined'''
            eri2c_K, eri3c_K = get_eri2c_eri3c_RSH(mol=mol, 
                                                auxmol=auxmol_K, 
                                                eri2c_K=eri2c_K, 
                                                eri3c_K=eri3c_K, 
                                                alpha=alpha, 
                                                beta=beta, 
                                                omega=omega, 
                                                single=single,
                                                max_mem_mb=max_mem_mb)

        lower_inv_eri2c_K = math_helper.matrix_power(eri2c_K,-0.5,epsilon=1e-6)
        lower_inv_eri2c_K = lower_inv_eri2c_K.astype(dtype=cp.float32 if single else cp.float64)

        unit = 4 if single else 8
        print(f'T_ia_K is going to take {  auxmol_K.nao_nr() * rest_occ * rest_vir * unit / (1024 ** 2):.0f} MB memory')
        print(f'T_ij_K is going to take {  auxmol_K.nao_nr() * rest_occ * rest_occ * unit / (1024 ** 2):.0f} MB memory')
        print(f'T_ab_K is going to take {  auxmol_K.nao_nr() * rest_vir * rest_vir * unit / (1024 ** 2):.0f} MB memory')
        

        T_ia_K = get_Tpq(eri3c=eri3c_K, lower_inv_eri2c=lower_inv_eri2c_K, C_p=C_occ_Ktrunc, C_q=C_vir_Ktrunc)
        T_ij_K = get_Tpq(eri3c=eri3c_K, lower_inv_eri2c=lower_inv_eri2c_K, C_p=C_occ_Ktrunc, C_q=C_occ_Ktrunc)
        T_ab_K = get_Tpq(eri3c=eri3c_K, lower_inv_eri2c=lower_inv_eri2c_K, C_p=C_vir_Ktrunc, C_q=C_vir_Ktrunc)

        T_ia_K = cp.asarray(T_ia_K)
        T_ij_K = cp.asarray(T_ij_K)
        T_ab_K = cp.asarray(T_ab_K)

        print(f'T_ia_K T_ij_K T_ab_K time {time.time() - tt:.1f} seconds')

        # hdiag = delta_hdiag.reshape(-1)
        hdiag_MVP = gen_hdiag_MVP(hdiag=hdiag, n_occ=n_occ, n_vir=n_vir)

        iajb_MVP = gen_iajb_MVP(T_left=T_ia_J, T_right=T_ia_J)
        ijab_MVP = gen_ijab_MVP(T_ij=T_ij_K,   T_ab=T_ab_K)
        ibja_MVP = get_ibja_MVP(T_ia=T_ia_K)

        def RKS_TDDFT_hybrid_MVP(X, Y):
            '''
            RKS
            [A B][X] = [AX+BY] = [U1]
            [B A][Y]   [AY+BX]   [U2]
            we want AX+BY and AY+BX
            instead of directly computing AX+BY and AY+BX 
            we compute (A+B)(X+Y) and (A-B)(X-Y)
            it can save one (ia|jb)V tensor contraction compared to directly computing AX+BY and AY+BX

            (A+B)V = hdiag_MVP(V) + 4*iajb_MVP(V) - a_x * [ ijab_MVP(V) + ibja_MVP(V) ]
            (A-B)V = hdiag_MVP(V) - a_x * [ ijab_MVP(V) - ibja_MVP(V) ]
            for RSH, a_x = 1, because the exchange component is defined by alpha+beta (alpha+beta not awlways == 1)

            # X Y in shape (m, n_occ*n_vir)
            '''
            tt = time.time()
            nstates = X.shape[0]
            X = X.reshape(nstates, n_occ, n_vir)
            Y = Y.reshape(nstates, n_occ, n_vir)

            XpY = X + Y
            XmY = X - Y
            # print('type(XpY)', type(XpY))
            ApB_XpY = hdiag_MVP(XpY) 
            tt = time.time()
            ApB_XpY += 4*iajb_MVP(XpY) 
            print(f'    iajb_MVP time {time.time() - tt:.2f}  seconds')
            tt = time.time()

            ApB_XpY[:,n_occ-rest_occ:,:rest_vir] -= a_x*ijab_MVP(XpY[:,n_occ-rest_occ:,:rest_vir]) 
            print(f'    ijab_MVP X+Y time {time.time() - tt:.2f}  seconds')
            tt = time.time()

            ApB_XpY[:,n_occ-rest_occ:,:rest_vir] -= a_x*ibja_MVP(XpY[:,n_occ-rest_occ:,:rest_vir])
            print(f'    ibja_MVP X+Y time {time.time() - tt:.2f}  seconds')
            tt = time.time()

            AmB_XmY = hdiag_MVP(XmY) 
            AmB_XmY[:,n_occ-rest_occ:,:rest_vir] -= a_x*ijab_MVP(XmY[:,n_occ-rest_occ:,:rest_vir]) 
            print(f'    ijab_MVP X-Y time {time.time() - tt:.2f}  seconds')
            tt = time.time()


            AmB_XmY[:,n_occ-rest_occ:,:rest_vir] += a_x*ibja_MVP(XmY[:,n_occ-rest_occ:,:rest_vir])
            print(f'    ibja_MVP X-Y time {time.time() - tt:.2f}  seconds')
            tt = time.time()

            ''' (A+B)(X+Y) = AX + BY + AY + BX   (1)
                (A-B)(X-Y) = AX + BY - AY - BX   (2)
                (1) + (1) /2 = AX + BY = U1
                (1) - (2) /2 = AY + BX = U2
            '''
            U1 = (ApB_XpY + AmB_XmY)/2
            U2 = (ApB_XpY - AmB_XmY)/2

            U1 = U1.reshape(nstates, n_occ*n_vir)
            U2 = U2.reshape(nstates, n_occ*n_vir)

            return U1, U2
        return RKS_TDDFT_hybrid_MVP, hdiag


    #  ===========  RKS pure ===========
    def get_RKS_TDA_pure_MVP(self):
        '''hybrid RKS TDA'''
        a_x = self.a_x
        n_occ = self.n_occ
        n_vir = self.n_vir

        mo_coeff = self.mf.mo_coeff
        mo_energy = self.mf.mo_energy

        single = self.single 

        C_occ_notrunc = self.C_occ_notrunc
        C_vir_notrunc = self.C_vir_notrunc

        rest_occ = self.rest_occ
        rest_vir = self.rest_vir

        hdiag = self.delta_hdiag.reshape(-1)

        mol = self.mol
        theta = self.theta 

        J_fit = self.J_fit

        max_mem_mb = self.max_mem_mb
        ''' RIJ '''
        tt = time.time()
        print('==================== RIJ ====================')
        auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)
        
        eri2c_J, eri3c_J = get_eri2c_eri3c(mol=mol, auxmol=auxmol_J, omega=0, single=single, max_mem_mb=max_mem_mb)

        lower_inv_eri2c_J = math_helper.matrix_power(eri2c_J,-0.5,epsilon=1e-6)
        lower_inv_eri2c_J = lower_inv_eri2c_J.astype(dtype=cp.float32 if single else cp.float64)
       
        T_ia_J = get_Tpq(eri3c=eri3c_J, lower_inv_eri2c=lower_inv_eri2c_J, C_p=C_occ_notrunc, C_q=C_vir_notrunc)
        print(f'T_ia_J time {time.time() - tt:.1f} seconds')
        tt = time.time()
 
        hdiag_MVP = gen_hdiag_MVP(hdiag=hdiag, n_occ=n_occ, n_vir=n_vir)
        iajb_MVP = gen_iajb_MVP(T_left=T_ia_J, T_right=T_ia_J)
        def RKS_TDA_pure_MVP(X):
            ''' pure functional, a_x = 0
                return AX
                AV = hdiag_MVP(V) + 2*iajb_MVP(V) 
            '''
            nstates = X.shape[0]
            X = X.reshape(nstates, n_occ, n_vir)
            AX = hdiag_MVP(X) 
            AX += 2 * iajb_MVP(X) 
            AX = AX.reshape(nstates, n_occ*n_vir)
            return AX

        return RKS_TDA_pure_MVP, hdiag
       
    def gen_RKS_TDDFT_pure_MVP(self):  
            
        a_x = self.a_x
        n_occ = self.n_occ
        n_vir = self.n_vir

        mo_coeff = self.mf.mo_coeff
        mo_energy = self.mf.mo_energy

        single = self.single 

        C_occ_notrunc = self.C_occ_notrunc
        C_vir_notrunc = self.C_vir_notrunc

        rest_occ = self.rest_occ
        rest_vir = self.rest_vir

        hdiag = self.delta_hdiag.reshape(-1)

        mol = self.mol
        theta = self.theta 

        J_fit = self.J_fit

        max_mem_mb = self.max_mem_mb
        ''' RIJ '''
        tt = time.time()
        print('==================== RIJ ====================')
        auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)
        
        eri2c_J, eri3c_J = get_eri2c_eri3c(mol=mol, auxmol=auxmol_J, omega=0, single=single, max_mem_mb=max_mem_mb)

        lower_inv_eri2c_J = math_helper.matrix_power(eri2c_J, -0.5, epsilon=1e-6)
        lower_inv_eri2c_J = lower_inv_eri2c_J.astype(dtype=cp.float32 if single else cp.float64)
       
        T_ia_J = get_Tpq(eri3c=eri3c_J, lower_inv_eri2c=lower_inv_eri2c_J, C_p=C_occ_notrunc, C_q=C_vir_notrunc)
        print(f'T_ia_J time {time.time() - tt:.1f} seconds')
        tt = time.time()
 
        hdiag_sqrt_MVP = gen_hdiag_MVP(hdiag=hdiag**0.5, n_occ=n_occ, n_vir=n_vir)
        hdiag_MVP = gen_hdiag_MVP(hdiag=hdiag, n_occ=n_occ, n_vir=n_vir)
        iajb_MVP = gen_iajb_MVP(T_left=T_ia_J, T_right=T_ia_J)
        hdiag_sq = hdiag**2
        def RKS_TDDFT_pure_MVP(Z):
            '''(A-B)^1/2(A+B)(A-B)^1/2 Z = Z w^2
                                    MZ = Z w^2
                M = (A-B)^1/2 (A+B) (A-B)^1/2
                X+Y = (A-B)^1/2 Z

                (A+B)(V) = hdiag_MVP(V) + 4*iajb_MVP(V)
                (A-B)^1/2(V) = hdiag_sqrt_MVP(V)
            '''
            nstates = Z.shape[0]
            Z = Z.reshape(nstates, n_occ, n_vir)
            AmB_sqrt_V = hdiag_sqrt_MVP(Z)
            ApB_AmB_sqrt_V = hdiag_MVP(AmB_sqrt_V) + 4*iajb_MVP(AmB_sqrt_V)
            MZ = hdiag_sqrt_MVP(ApB_AmB_sqrt_V)
            MZ = MZ.reshape(nstates, n_occ*n_vir)
            return MZ
        
        return RKS_TDDFT_pure_MVP, hdiag_sq
        
    #  ===========  UKS ===========
    def get_UKS_TDA_MVP(self):
        a_x = self.a_x

        n_occ_a = self.n_occ_a
        n_vir_a = self.n_vir_a
        n_occ_b = self.n_occ_b
        n_vir_b = self.n_vir_b      

        A_aa_size = n_occ_a * n_vir_a
        A_bb_size = n_occ_b * n_vir_b

        mo_coeff = self.mf.mo_coeff
        mo_energy = self.mf.mo_energy

        mol = self.mol
        auxmol = self.get_auxmol(theta=self.theta, add_p=self.add_p)
        eri2c, eri3c = self.get_eri2c_eri3c(mol=self.mol, auxmol=auxmol, omega=0)
        uvP_withL = self.get_uvP_withL(eri2c=eri2c, eri3c=eri3c)

        hdiag_a_MVP, hdiag_a = self.get_hdiag_MVP(mo_energy=mo_energy[0], n_occ=n_occ_a, n_vir=n_vir_a)
        hdiag_b_MVP, hdiag_b = self.get_hdiag_MVP(mo_energy=mo_energy[1], n_occ=n_occ_b, n_vir=n_vir_b)
        hdiag = cp.vstack((hdiag_a.reshape(-1,1), hdiag_b.reshape(-1,1))).reshape(-1)

        if a_x != 0:
            ''' UKS TDA hybrid '''
            T_ia_J_alpha, _, T_ij_K_alpha, T_ab_K_alpha = self.get_T_J_T_K(mol=mol,
                                                                                auxmol=auxmol,
                                                                                uvP_withL=uvP_withL, 
                                                                                eri3c=eri3c, 
                                                                                eri2c=eri2c,
                                                                                n_occ=n_occ_a, 
                                                                                mo_coeff=mo_coeff[0])
            
            T_ia_J_beta, _, T_ij_K_beta, T_ab_K_beta  = self.get_T_J_T_K(mol=mol,
                                                                              auxmol=auxmol,
                                                                              uvP_withL=uvP_withL,
                                                                              eri3c=eri3c, 
                                                                              eri2c=eri2c,
                                                                              n_occ=n_occ_b,
                                                                              mo_coeff=mo_coeff[1])

            iajb_aa_MVP = self.gen_iajb_MVP(T_left=T_ia_J_alpha, T_right=T_ia_J_alpha)
            iajb_ab_MVP = self.gen_iajb_MVP(T_left=T_ia_J_alpha, T_right=T_ia_J_beta)
            iajb_ba_MVP = self.gen_iajb_MVP(T_left=T_ia_J_beta,  T_right=T_ia_J_alpha)
            iajb_bb_MVP = self.gen_iajb_MVP(T_left=T_ia_J_beta,  T_right=T_ia_J_beta)

            ijab_aa_MVP = self.gen_ijab_MVP(T_ij=T_ij_K_alpha, T_ab=T_ab_K_alpha)
            ijab_bb_MVP = self.gen_ijab_MVP(T_ij=T_ij_K_beta,  T_ab=T_ab_K_beta)
            
            def UKS_TDA_hybrid_MVP(X):
                '''
                UKS
                return AX
                A have 4 blocks, αα, αβ, βα, ββ
                A = [ Aαα Aαβ ]  
                    [ Aβα Aββ ]       

                X = [ Xα ]   
                    [ Xβ ]     
                AX = [ Aαα Xα + Aαβ Xβ ]
                     [ Aβα Xα + Aββ Xβ ]

                Aαα Xα = hdiag_MVP(Xα) + iajb_aa_MVP(Xα) - a_x * ijab_aa_MVP(Xα) 
                Aββ Xβ = hdiag_MVP(Xβ) + iajb_bb_MVP(Xβ) - a_x * ijab_bb_MVP(Xβ)
                Aαβ Xβ = iajb_ab_MVP(Xβ)
                Aβα Xα = iajb_ba_MVP(Xα)
                '''
                X_a = X[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                X_b = X[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)

                Aaa_Xa = hdiag_a_MVP(X_a) + iajb_aa_MVP(X_a) - a_x * ijab_aa_MVP(X_a) 
                Aab_Xb = iajb_ab_MVP(X_b)

                Aba_Xa = iajb_ba_MVP(X_a)
                Abb_Xb = hdiag_b_MVP(X_b) + iajb_bb_MVP(X_b) - a_x * ijab_bb_MVP(X_b)
                
                U_a = (Aaa_Xa + Aab_Xb).reshape(A_aa_size,-1)
                U_b = (Aba_Xa + Abb_Xb).reshape(A_bb_size,-1)

                U = cp.vstack((U_a, U_b))
                return U
            return UKS_TDA_hybrid_MVP, hdiag

        elif a_x == 0:
            ''' UKS TDA pure '''
            T_ia_alpha = self.get_T(uvP_withL=uvP_withL,
                                    n_occ=n_occ_a, 
                                    mo_coeff=mo_coeff[0],
                                    calc='coulomb_only')
            T_ia_beta = self.get_T(uvP_withL=uvP_withL,
                                    n_occ=n_occ_b, 
                                    mo_coeff=mo_coeff[1],
                                    calc='coulomb_only')
            
            iajb_aa_MVP = self.gen_iajb_MVP(T_left=T_ia_alpha, T_right=T_ia_alpha)
            iajb_ab_MVP = self.gen_iajb_MVP(T_left=T_ia_alpha, T_right=T_ia_beta)
            iajb_ba_MVP = self.gen_iajb_MVP(T_left=T_ia_beta,  T_right=T_ia_alpha)
            iajb_bb_MVP = self.gen_iajb_MVP(T_left=T_ia_beta,  T_right=T_ia_beta)

            def UKS_TDA_pure_MVP(X):
                '''
                Aαα Xα = hdiag_MVP(Xα) + iajb_aa_MVP(Xα)  
                Aββ Xβ = hdiag_MVP(Xβ) + iajb_bb_MVP(Xβ) 
                Aαβ Xβ = iajb_ab_MVP(Xβ)
                Aβα Xα = iajb_ba_MVP(Xα)
                '''
                X_a = X[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                X_b = X[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)

                Aaa_Xa = hdiag_a_MVP(X_a) + iajb_aa_MVP(X_a)
                Aab_Xb = iajb_ab_MVP(X_b)

                Aba_Xa = iajb_ba_MVP(X_a)
                Abb_Xb = hdiag_b_MVP(X_b) + iajb_bb_MVP(X_b) 
                
                U_a = (Aaa_Xa + Aab_Xb).reshape(A_aa_size,-1)
                U_b = (Aba_Xa + Abb_Xb).reshape(A_bb_size,-1)

                U = cp.vstack((U_a, U_b))
                return U
            return UKS_TDA_pure_MVP, hdiag

    def get_UKS_TDDFT_MVP(self):
        
        a_x = self.a_x

        n_occ_a = self.n_occ_a
        n_vir_a = self.n_vir_a
        n_occ_b = self.n_occ_b
        n_vir_b = self.n_vir_b      

        A_aa_size = n_occ_a * n_vir_a
        A_bb_size = n_occ_b * n_vir_b

        mo_coeff = self.mf.mo_coeff
        mo_energy = self.mf.mo_energy

        '''
        the 2c2e and 3c2e integrals with/without RSH
        (ij|ab) = (ij|1-(alpha + beta*erf(omega))/r|ab)  + (ij|alpha + beta*erf(omega)/r|ab)
        short-range part (ij|1-(alpha + beta*erf(omega))/r|ab) is treated by the DFT XC functional, thus not considered here
        long-range part  (ij|alpha + beta*erf(omega)/r|ab) = alpha (ij|r|ab) + beta*(ij|erf(omega)/r|ab)
        ''' 
        mol = self.mol
        auxmol = self.get_auxmol(theta=self.theta, add_p=self.add_p)
        eri2c, eri3c = self.get_eri2c_eri3c(mol=self.mol, auxmol=auxmol, omega=0)
        uvP_withL = self.get_uvP_withL(eri2c=eri2c, eri3c=eri3c)
        '''
        _aa_MVP means alpha-alpha spin
        _ab_MVP means alpha-beta spin
        T_ia_alpha means T_ia matrix for alpha spin
        T_ia_beta means T_ia matrix for beta spin
        '''

        hdiag_a_MVP, hdiag_a = self.get_hdiag_MVP(mo_energy=mo_energy[0], n_occ=n_occ_a, n_vir=n_vir_a)
        hdiag_b_MVP, hdiag_b = self.get_hdiag_MVP(mo_energy=mo_energy[1], n_occ=n_occ_b, n_vir=n_vir_b)
        hdiag = cp.vstack((hdiag_a.reshape(-1,1), hdiag_b.reshape(-1,1))).reshape(-1)

        if a_x != 0:
            T_ia_J_alpha, T_ia_K_alpha, T_ij_K_alpha, T_ab_K_alpha = self.get_T_J_T_K(mol=mol,
                                                                                            auxmol=auxmol,
                                                                                            uvP_withL=uvP_withL, 
                                                                                            eri3c=eri3c, 
                                                                                            eri2c=eri2c,
                                                                                            n_occ=n_occ_a, 
                                                                                            mo_coeff=mo_coeff[0])
            
            T_ia_J_beta,  T_ia_K_beta,  T_ij_K_beta,  T_ab_K_beta  = self.get_T_J_T_K(mol=mol,
                                                                                            auxmol=auxmol,
                                                                                            uvP_withL=uvP_withL,
                                                                                            eri3c=eri3c, 
                                                                                            eri2c=eri2c,
                                                                                            n_occ=n_occ_b,
                                                                                            mo_coeff=mo_coeff[1])

            iajb_aa_MVP = self.gen_iajb_MVP(T_left=T_ia_J_alpha, T_right=T_ia_J_alpha)
            iajb_ab_MVP = self.gen_iajb_MVP(T_left=T_ia_J_alpha, T_right=T_ia_J_beta)
            iajb_ba_MVP = self.gen_iajb_MVP(T_left=T_ia_J_beta,  T_right=T_ia_J_alpha)
            iajb_bb_MVP = self.gen_iajb_MVP(T_left=T_ia_J_beta,  T_right=T_ia_J_beta)

            ijab_aa_MVP = self.gen_ijab_MVP(T_ij=T_ij_K_alpha, T_ab=T_ab_K_alpha)
            ijab_bb_MVP = self.gen_ijab_MVP(T_ij=T_ij_K_beta,  T_ab=T_ab_K_beta)
            
            ibja_aa_MVP = self.get_ibja_MVP(T_ia=T_ia_K_alpha)
            ibja_bb_MVP = self.get_ibja_MVP(T_ia=T_ia_K_beta)

            def UKS_TDDFT_hybrid_MVP(X,Y):
                '''
                UKS
                [A B][X] = [AX+BY] = [U1]
                [B A][Y]   [AY+BX]   [U2]
                A B have 4 blocks, αα, αβ, βα, ββ
                A = [ Aαα Aαβ ]   B = [ Bαα Bαβ ]
                    [ Aβα Aββ ]       [ Bβα Bββ ]

                X = [ Xα ]        Y = [ Yα ]
                    [ Xβ ]            [ Yβ ]

                (A+B)αα, (A+B)αβ is shown below

                βα, ββ can be obtained by change α to β 
                we compute (A+B)(X+Y) and (A-B)(X-Y)

                V:= X+Y
                (A+B)αα Vα = hdiag_MVP(Vα) + 2*iaαjbα_MVP(Vα) - a_x*[ijαabα_MVP(Vα) + ibαjaα_MVP(Vα)]
                (A+B)αβ Vβ = 2*iaαjbβ_MVP(Vβ) 

                V:= X-Y
                (A-B)αα Vα = hdiag_MVP(Vα) - a_x*[ijαabα_MVP(Vα) - ibαjaα_MVP(Vα)]
                (A-B)αβ Vβ = 0

                A+B = [ Cαα Cαβ ]   x+y = [ Vα ]  
                      [ Cβα Cββ ]         [ Vβ ]
                (A+B)(x+y) =   [ Cαα Vα + Cαβ Vβ ]  = ApB_XpY
                               [ Cβα Vα + Cββ Vβ ]

                A-B = [ Cαα  0  ]   x-y = [ Vα ]   
                      [  0  Cββ ]         [ Vβ ]                          
                (A-B)(x-y) =   [ Cαα Vα ]    = AmB_XmY
                               [ Cββ Vβ ]
                '''
                
                X_a = X[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                X_b = X[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)
                Y_a = Y[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                Y_b = Y[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)
                
                XpY_a = X_a + Y_a
                XpY_b = X_b + Y_b

                XmY_a = X_a - Y_a
                XmY_b = X_b - Y_b

                '''============== (A+B) (X+Y) ================'''
                '''(A+B)aa(X+Y)a'''
                ApB_XpY_aa = hdiag_a_MVP(XpY_a) + 2*iajb_aa_MVP(XpY_a) - a_x*(ijab_aa_MVP(XpY_a) + ibja_aa_MVP(XpY_a))
                '''(A+B)bb(X+Y)b'''
                ApB_XpY_bb = hdiag_b_MVP(XpY_b) + 2*iajb_bb_MVP(XpY_b) - a_x*(ijab_bb_MVP(XpY_b) + ibja_bb_MVP(XpY_b))           
                '''(A+B)ab(X+Y)b'''
                ApB_XpY_ab = 2*iajb_ab_MVP(XpY_b)
                '''(A+B)ba(X+Y)a'''
                ApB_XpY_ba = 2*iajb_ba_MVP(XpY_a)

                '''============== (A-B) (X-Y) ================'''
                '''(A-B)aa(X-Y)a'''
                AmB_XmY_aa = hdiag_a_MVP(XmY_a) - a_x*(ijab_aa_MVP(XmY_a) - ibja_aa_MVP(XmY_a))
                '''(A-B)bb(X-Y)b'''
                AmB_XmY_bb = hdiag_b_MVP(XmY_b) - a_x*(ijab_bb_MVP(XmY_b) - ibja_bb_MVP(XmY_b))  

                ''' (A-B)ab(X-Y)b
                    AmB_XmY_ab = 0
                    (A-B)ba(X-Y)a
                    AmB_XmY_ba = 0
                '''

                ''' (A+B)(X+Y) = AX + BY + AY + BX   (1) ApB_XpY
                    (A-B)(X-Y) = AX + BY - AY - BX   (2) AmB_XmY
                    (1) + (1) /2 = AX + BY = U1
                    (1) - (2) /2 = AY + BX = U2
                '''
                ApB_XpY_alpha = (ApB_XpY_aa + ApB_XpY_ab).reshape(A_aa_size,-1)
                ApB_XpY_beta  = (ApB_XpY_ba + ApB_XpY_bb).reshape(A_bb_size,-1)
                ApB_XpY = cp.vstack((ApB_XpY_alpha, ApB_XpY_beta))

                AmB_XmY_alpha = AmB_XmY_aa.reshape(A_aa_size,-1)
                AmB_XmY_beta  = AmB_XmY_bb.reshape(A_bb_size,-1)
                AmB_XmY = cp.vstack((AmB_XmY_alpha, AmB_XmY_beta))
                
                U1 = (ApB_XpY + AmB_XmY)/2
                U2 = (ApB_XpY - AmB_XmY)/2

                return U1, U2

            return UKS_TDDFT_hybrid_MVP, hdiag

        elif a_x == 0:
            ''' UKS TDDFT pure '''

            hdiag_a_sqrt_MVP, hdiag_a_sq = self.get_hdiag_MVP(mo_energy=mo_energy[0], 
                                                          n_occ=n_occ_a, 
                                                          n_vir=n_vir_a,
                                                          sqrt=True)
            hdiag_b_sqrt_MVP, hdiag_b_sq = self.get_hdiag_MVP(mo_energy=mo_energy[1], 
                                                          n_occ=n_occ_b, 
                                                          n_vir=n_vir_b,
                                                          sqrt=True)
            '''hdiag_sq: preconditioner'''
            hdiag_sq = cp.vstack((hdiag_a_sq.reshape(-1,1), hdiag_b_sq.reshape(-1,1))).reshape(-1)

            T_ia_alpha = self.get_T(uvP_withL=uvP_withL,
                                    n_occ=n_occ_a, 
                                    mo_coeff=mo_coeff[0],
                                    calc='coulomb_only')
            T_ia_beta = self.get_T(uvP_withL=uvP_withL,
                                    n_occ=n_occ_b, 
                                    mo_coeff=mo_coeff[1],
                                    calc='coulomb_only')
                       
            iajb_aa_MVP = self.gen_iajb_MVP(T_left=T_ia_alpha, T_right=T_ia_alpha)
            iajb_ab_MVP = self.gen_iajb_MVP(T_left=T_ia_alpha, T_right=T_ia_beta)
            iajb_ba_MVP = self.gen_iajb_MVP(T_left=T_ia_beta,  T_right=T_ia_alpha)
            iajb_bb_MVP = self.gen_iajb_MVP(T_left=T_ia_beta,  T_right=T_ia_beta)

            def UKS_TDDFT_pure_MVP(Z):
                '''       MZ = Z w^2
                    M = (A-B)^1/2(A+B)(A-B)^1/2
                    Z = (A-B)^1/2(X-Y)

                    X+Y = (A-B)^1/2 Z * 1/w
                    A+B = hdiag_MVP(V) + 4*iajb_MVP(V)
                    (A-B)^1/2 = hdiag_sqrt_MVP(V)


                    M =  [ (A-B)^1/2αα    0   ] [ (A+B)αα (A+B)αβ ] [ (A-B)^1/2αα    0   ]            Z = [ Zα ]  
                         [    0   (A-B)^1/2ββ ] [ (A+B)βα (A+B)ββ ] [    0   (A-B)^1/2ββ ]                [ Zβ ]
                '''
                Z_a = Z[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                Z_b = Z[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)

                AmB_aa_sqrt_Z_a = hdiag_a_sqrt_MVP(Z_a)
                AmB_bb_sqrt_Z_b = hdiag_b_sqrt_MVP(Z_b)

                ApB_aa_sqrt_V = hdiag_a_MVP(AmB_aa_sqrt_Z_a) + 2*iajb_aa_MVP(AmB_aa_sqrt_Z_a)
                ApT_ab_sqrt_V = 2*iajb_ab_MVP(AmB_bb_sqrt_Z_b)
                ApB_ba_sqrt_V = 2*iajb_ba_MVP(AmB_aa_sqrt_Z_a)
                ApB_bb_sqrt_V = hdiag_b_MVP(AmB_bb_sqrt_Z_b) + 2*iajb_bb_MVP(AmB_bb_sqrt_Z_b)

                MZ_a = hdiag_a_sqrt_MVP(ApB_aa_sqrt_V + ApT_ab_sqrt_V).reshape(A_aa_size, -1)
                MZ_b = hdiag_b_sqrt_MVP(ApB_ba_sqrt_V + ApB_bb_sqrt_V).reshape(A_bb_size, -1)

                MZ = cp.vstack((MZ_a, MZ_b))
                # print(MZ.shape)
                return MZ
        
            return UKS_TDDFT_pure_MVP, hdiag_sq

        # def TDDFT_spolar_MVP(X):

        #     ''' for RSH, a_x=1
        #         (A+B)X = hdiag_MVP(V) + 4*iajb_MVP(V) - a_x*[ijab_MVP(V) + ibja_MVP(V)]
        #     '''
        #     X = X.reshape(n_occ, n_vir, -1)

        #     ABX = hdiag_MVP(X) + 4*iajb_MVP(X) - a_x* (ibja_MVP(X) + ijab_MVP(X))
        #     ABX = ABX.reshape(n_occ*n_vir, -1)

        #     return ABX

    def get_inter_contract_C(self, int_tensor, mo_coeff, mo_occ):
        '''
        transition dipole
        mol: mol obj
        '''
        occidx = mo_occ > 0
        orbo = mo_coeff[:, occidx]
        orbv = mo_coeff[:,~occidx]

        # P = einsum("xpq,pi,qa->xia", int_tensor, orbo, orbv.conj())
        P = get_pre_Tpq_one_batch(int_tensor, orbo, orbv.conj())

        P = cp.asarray(P.reshape(3,-1))

        return P

    def get_RKS_P(self):
        '''
        transition dipole u
        '''
        tag = self.eri_tag
        int_r = self.mol.intor_symmetric('int1e_r'+tag)
        int_r = np.asarray(int_r, dtype=cp.float32 if self.single else cp.float64)
        mo_occ = self.mf.mo_occ
        mo_coeff = self.mf.mo_coeff

        P = self.get_inter_contract_C(int_tensor=int_r, mo_coeff=mo_coeff, mo_occ=mo_occ)
        # P = P.reshape(-1,3)
        return P

    def get_RKS_mdpol(self):
        '''
        magnatic dipole m
        '''
        tag = self.eri_tag
        # int_rxp = self.mol.intor_symmetric('int1e_cg_irxp'+tag)
        int_rxp = self.mol.intor('int1e_cg_irxp', comp=3, hermi=2)
        int_rxp = np.asarray(int_rxp, dtype=cp.float32 if self.single else cp.float64)
        mo_occ = self.mf.mo_occ
        mo_coeff = self.mf.mo_coeff

        mdpol = self.get_inter_contract_C(int_tensor=int_rxp, mo_coeff=mo_coeff, mo_occ=mo_occ)
        # P = P.reshape(-1,3)
        return mdpol

    def get_UKS_P(self):
        '''
        transition dipole,
        P = [P_α]
            [P_β]
        '''
        tag = self.eri_tag
        int_r = self.mol.intor_symmetric('int1e_r'+tag)

        mo_occ = self.mf.mo_occ
        mo_coeff = self.mf.mo_coeff

        P_alpha = self.get_inter_contract_C(int_r=int_r, mo_coeff=mo_coeff[0], mo_occ=mo_occ[0])
        P_beta = self.get_inter_contract_C(int_r=int_r, mo_coeff=mo_coeff[1], mo_occ=mo_occ[1])
        P = cp.vstack((P_alpha, P_beta))
        return P

    def kernel_TDA(self):
 
        '''for TDA, pure and hybrid share the same form of
                     AX = Xw
            always use the Davidson solver
            pure TDA is not using MZ=Zw^2 form
        '''

        if self.RKS:
            P = self.get_RKS_P()
            mdpol = self.get_RKS_mdpol()

            if self.a_x != 0:
                TDA_MVP, hdiag = self.get_RKS_TDA_hybrid_MVP()

            elif self.a_x == 0:
                TDA_MVP, hdiag = self.get_RKS_TDA_pure_MVP()


        elif self.UKS:
            TDA_MVP, hdiag = self.get_UKS_TDA_MVP()
            P = self.get_UKS_P()

        energies, X = eigen_solver.Davidson(matrix_vector_product=TDA_MVP,
                                            hdiag=hdiag,
                                            N_states=self.nroots,
                                            conv_tol=self.conv_tol,
                                            max_iter=self.max_iter,
                                            GS=self.GS,
                                            single=self.single)
        Xnorm = cp.linalg.norm(cp.dot(X, X.T) - cp.eye(X.shape[0]))
        print(f'check orthonormal of X: {Xnorm:.2e}')

        oscillator_strength = spectralib.get_spectra(energies=energies, 
                                                       X=X/(2**0.5),
                                                       Y=None,
                                                       P=P, 
                                                       mdpol=mdpol,
                                                       name=self.out_name+'_TDA_ris', 
                                                       RKS=self.RKS,
                                                       spectra=self.spectra,
                                                       print_threshold = self.print_threshold,
                                                       n_occ=self.n_occ if self.RKS else (self.n_occ_a, self.n_occ_b),
                                                       n_vir=self.n_vir if self.RKS else (self.n_vir_a, self.n_vir_b))
        energies = energies*parameter.Hartree_to_eV


        return energies, X, oscillator_strength

    def kernel_TDDFT(self):     
        # math_helper.show_memory_info('At the beginning')
        if self.a_x != 0:
            '''hybrid TDDFT'''
            if self.RKS:
                P = self.get_RKS_P()
                mdpol = self.get_RKS_mdpol()
                TDDFT_hybrid_MVP, hdiag = self.gen_RKS_TDDFT_hybrid_MVP()

            elif self.UKS:
                P = self.get_UKS_P()
                TDDFT_hybrid_MVP, hdiag = self.get_UKS_TDDFT_MVP()

            energies, X, Y = eigen_solver.Davidson_Casida(matrix_vector_product=TDDFT_hybrid_MVP, 
                                                            hdiag=hdiag,
                                                            N_states=self.nroots,
                                                            conv_tol=self.conv_tol,
                                                            max_iter=self.max_iter,
                                                            GS=self.GS,
                                                            single=self.single)

        elif self.a_x == 0:
            '''pure TDDFT'''
            if self.RKS:
                TDDFT_pure_MVP, hdiag_sq = self.gen_RKS_TDDFT_pure_MVP()
                P = self.get_RKS_P()
                mdpol = self.get_RKS_mdpol()

            elif self.UKS:
                TDDFT_pure_MVP, hdiag_sq = self.get_UKS_TDDFT_MVP()
                P = self.get_UKS_P()
            # print('min(hdiag_sq)', min(hdiag_sq)**0.5*parameter.Hartree_to_eV)
            energies_sq, Z = eigen_solver.Davidson(matrix_vector_product=TDDFT_pure_MVP, 
                                                hdiag=hdiag_sq,
                                                N_states=self.nroots,
                                                conv_tol=self.conv_tol,
                                                max_iter=self.max_iter,
                                                GS=self.GS,
                                                single=self.single)
            
            energies = energies_sq**0.5
            Z = (energies**0.5).reshape(-1,1) * Z

            X, Y = math_helper.XmY_2_XY(Z=Z, AmB_sq=hdiag_sq, omega=energies)

        XY_norm_check = cp.linalg.norm( (cp.dot(X, X.T) - cp.dot(Y, Y.T)) - cp.eye(self.nroots) )
        print(f'check norm of X^TX - Y^YY - I = {XY_norm_check:.2e}')

    
        oscillator_strength = spectralib.get_spectra(energies=energies, 
                                                    X=X/(2**0.5),
                                                    Y=Y/(2**0.5),
                                                    P=P, 
                                                    mdpol=mdpol,
                                                    name=self.out_name+'_TDDFT_ris', 
                                                    spectra=self.spectra,
                                                    RKS=self.RKS,
                                                    print_threshold = self.print_threshold,
                                                    n_occ=self.n_occ if self.RKS else (self.n_occ_a, self.n_occ_b),
                                                    n_vir=self.n_vir if self.RKS else (self.n_vir_a, self.n_vir_b))
        energies = energies*parameter.Hartree_to_eV


        return energies, X, Y, oscillator_strength

