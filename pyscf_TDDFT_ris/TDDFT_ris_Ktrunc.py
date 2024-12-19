from pyscf import gto, lib, dft
import scipy
import numpy as np
import multiprocessing as mp
from pyscf_TDDFT_ris import parameter, eigen_solver, math_helper, spectralib
import gc


np.set_printoptions(linewidth=250, threshold=np.inf)

einsum = lib.einsum


num_cores = int(mp.cpu_count())
print("This job can use: " + str(num_cores) + "CPUs")


def print_memory_usage(line):
    import psutil
    process = psutil.Process()

    memory_usage = process.memory_info().rss  # rss: 常驻内存大小 (单位：字节)

    print(f"memory usage at {line}: {memory_usage / (1024 ** 2):.0f} MB")

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
    auxmol = gto.M(atom=mol.atom, 
                    parse_arg = False, 
                    spin=mol.spin, 
                    charge=mol.charge,
                    cart=mol.cart)
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

    # print('=====================')
    # [print(k, v) for k, v in auxmol._basis.items()]
    return auxmol


def get_eri2c_eri3c(mol, auxmol, omega=0, single=True):

    '''
    Total number of contracted GTOs for the mole and auxmol object
    '''
    nbf = mol.nao_nr()
    nauxbf = auxmol.nao_nr()

    predict_mem = nbf * nbf * nauxbf * 4 / (1024 ** 2)
    print(f"Predicted memory usage for 3c2e: {predict_mem:.2f} MB")

    if omega != 0:
        mol.set_range_coulomb(omega)
        auxmol.set_range_coulomb(omega)

    '''
    (pq|rs) = Σ_PQ (pq|P)(P|Q)^-1(Q|rs)
    2 center 2 electron integral (P|Q)
    N_auxbf * N_auxbf
    '''

    tag = '_cart' if mol.cart else '_sph'

    eri2c = auxmol.intor('int2c2e' + tag)
    # print(eri2c)
    '''
    3 center 2 electron integral (pq|P)
    N_bf * N_bf * N_auxbf
    '''
    pmol = mol + auxmol
    pmol.cart = mol.cart
    print('auxmol.cart =', mol.cart)

    shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + auxmol.nbas)
    eri3c = pmol.intor('int3c2e' + tag, shls_slice=shls_slice)
    # print('mol.shls_slice =', shls_slice)

    if single:
        eri2c = eri2c.astype(np.float32)
        gc.collect()

        eri3c = eri3c.astype(np.float32) 
        gc.collect()
    print(f"eri3c memory usage:{eri3c.nbytes / (1024 ** 2):.2f} MB")
    
    print('Three center ERI shape', eri3c.shape)

    # del eri3c
    # print_memory_usage('after del eri3c')

    # import h5py
    # ftmp = lib.H5TmpFile()
    # eri3c_tmp = ftmp.create_dataset('eri3c', shape=eri3c.shape, dtype='float32')
    # eri3c_tmp[:] = eri3c
    # with h5py.File('eri3c.h5', 'w') as h5file:
    #     eri3c_dataset = h5file.create_dataset(
    #         'eri3c',
    #         shape=(nbf, nbf, nauxbf),
    #         dtype='float32',
    #         compression='gzip'
    #     )

    #     # 按辅助基函数的维度分块写入
    #     block_size = 50
    #     for aux_start in range(0, nauxbf, block_size):
    #         aux_end = min(aux_start + block_size, nauxbf)
    #         eri3c_dataset[:, :, aux_start:aux_end] = eri3c[:, :, aux_start:aux_end]
    #         print(f"Written block {aux_start}:{aux_end}")

    return eri2c, eri3c

def get_eri2c_eri3c_RSH(mol, auxmol, eri2c_K, eri3c_K, alpha, beta, omega, single=False):

    '''
    in the RSH functional, the Exchange ERI splits into two parts
    (ij|ab) = (ij|1-(alpha + beta*erf(omega))/r|ab) + (ij|alpha + beta*erf(omega)/r|ab)
    -- The first part (ij|1-(alpha + beta*erf(omega))/r|ab) is short range, 
        treated by the DFT XC functional, thus not considered here
    -- The second part is long range 
        (ij|alpha + beta*erf(omega)/r|ab) = alpha (ij|r|ab) + beta*(ij|erf(omega)/r|ab)
    '''            
    print('2c2e and 3c2e for RSH RI-K (ij|ab)')
    eri2c_erf, eri3c_erf = get_eri2c_eri3c(mol=mol, auxmol=auxmol, omega=omega, single=single)
    eri2c_RSH = alpha * eri2c_K + beta * eri2c_erf
    gc.collect() 

    eri3c_RSH = alpha * eri3c_K + beta * eri3c_erf
    gc.collect()    
    return eri2c_RSH, eri3c_RSH

def get_uvP_withL(eri3c, eri2c):
    '''
    (P|Q)^-1 = LL^T
    uvP_withL = Σ_P (uv|P)L_P
    '''
    # print('eri3c.shape', eri3c.shape)
    Lower = np.linalg.cholesky(np.linalg.inv(eri2c))
    print_memory_usage('Lower')
    print(f"Lower memory usage:{Lower.nbytes / (1024 ** 2):.2f} MB")
    print('eri3c.dtype', eri3c.dtype)
    # uvP_withL = einsum("uvQ,QP->uvP", eri3c, Lower)
    nbf, nbf, nauxbf = eri3c.shape

    eri3c = eri3c.reshape(nbf*nbf, nauxbf)
    uvP_withL = np.dot(eri3c, Lower.T)
    del eri3c
    uvP_withL = uvP_withL.reshape(nbf, nbf, Lower.shape[1])

    print_memory_usage('uvP_withL')
    print(f"uvP_withL memory usage:{uvP_withL.nbytes / (1024 ** 2):.2f} MB")
    return uvP_withL

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

# def get_Tia(uvP_withL: np.ndarray, C_occ: np.ndarray, C_vir: np.ndarray):
#     '''    
#     T means rank-3 Tensor
#     T_pq^P = C_up Σ_Q (uv|Q)L_Q  C_vq 
#     C_occ: C[:, :n_occ]
#     C_vir: C[:, n_occ:]
#     uvP_withL: Σ_P (uv|P)L_P
#     '''
#     tmp = einsum("va,uvP->uaP", C_vir, uvP_withL)
#     T_ia = einsum("ui,uaP->iaP", C_occ, tmp)
#     del uvP_withL, tmp
#     print('T_ia.shape', T_ia.shape)
#     return T_ia 

def get_Tia(eri3c: np.ndarray, lower_inv_eri2c: np.ndarray, C_occ: np.ndarray, C_vir: np.ndarray):
    '''    
    T means rank-3 Tensor

    T_pq = Σ_uvQ C_up (uv|Q)L_Q  C_vq

    preT_pq^P =  Σ_uv C_up (uv|Q) C_vq 

    C_occ: C[:, :n_occ]
    C_vir: C[:, n_occ:]

    lower_inv_eri2c (L_Q): (P|Q)^-1 = LL^T

    '''

    tmp = einsum("ui,uvP->ivP", C_occ, eri3c)
    pre_T_ia = einsum("va,ivP->iaP", C_vir, tmp)

    T_ia = einsum("iaP,PQ->iaQ", pre_T_ia, lower_inv_eri2c)
    # del uvP_withL, tmp
    print('T_ia.shape', T_ia.shape)
    return T_ia 


# def get_Tij_Tab(uvP_withL: np.ndarray, C_occ: np.ndarray, C_vir: np.ndarray):
#     '''
#     For common bybrid DFT, exchange and coulomb term use same set of T matrix
#     For range-seperated bybrid DFT, (ij|ab) and (ib|ja) use different T matrix than (ia|jb), 
#     because of the RSH eri2c and eri3c.
#     T_ia_K is only for (ib|ja)
#     '''

#     T_ij = einsum("ui,vj,uvP->ijP", C_occ, C_occ, uvP_withL)
#     T_ab = einsum("ua,vb,uvP->abP", C_vir, C_vir, uvP_withL)
#     del uvP_withL
#     print('T_ij.shape', T_ij.shape)
#     print('T_ab.shape', T_ab.shape)
#     return T_ij, T_ab


def get_Tij_Tab(eri3c: np.ndarray, lower_inv_eri2c: np.ndarray, C_occ: np.ndarray, C_vir: np.ndarray):
    '''
    For common bybrid DFT, exchange and coulomb term use same set of T matrix
    For range-seperated bybrid DFT, (ij|ab) and (ib|ja) use different T matrix than (ia|jb), 
    because of the RSH eri2c and eri3c.
    T_ia_K is only for (ib|ja)
    '''

    tmp = einsum("ui,uvP->ivP", C_occ, eri3c)
    pre_T_ij = einsum("vj,ivP->ijP", C_occ, tmp)
    T_ij = einsum("ijP,PQ->ijQ", pre_T_ij, lower_inv_eri2c)

    tmp = einsum("ua,uvP->avP", C_vir, eri3c)
    pre_T_ab = einsum("vb,avP->abP", C_vir, tmp)
    T_ab = einsum("abP,PQ->abQ", pre_T_ab, lower_inv_eri2c)


    print('T_ij.shape', T_ij.shape)
    print('T_ab.shape', T_ab.shape)
    return T_ij, T_ab

def gen_hdiag_MVP(mo_energy, n_occ, n_vir, sqrt=False):

    '''KS orbital energy difference, ε_a - ε_i
    '''
    vir = mo_energy[n_occ:].reshape(1,n_vir)
    occ = mo_energy[:n_occ].reshape(n_occ,1)
    delta_hdiag = np.repeat(vir, n_occ, axis=0) - np.repeat(occ, n_vir, axis=1)

    hdiag = delta_hdiag.reshape(n_occ*n_vir)
    # print(hdiag)
    # print('delta_hdiag[-1,0]', delta_hdiag[-1,0]*parameter.Hartree_to_eV)
    # Hdiag = np.vstack((hdiag, hdiag))
    # Hdiag = Hdiag.reshape(-1)
    if sqrt == False:
        '''standard diag(A)V
            preconditioner = diag(A)
        '''
        def hdiag_MVP(V):
            delta_hdiag_v = einsum("ia,iam->iam", delta_hdiag, V)
            return delta_hdiag_v
        return hdiag_MVP, hdiag
    
    elif sqrt == True:
        '''diag(A)**0.5 V
            preconditioner = diag(A)**2
        '''
        delta_hdiag_sqrt = np.sqrt(delta_hdiag)
        hdiag_sq = hdiag**2
        def hdiag_sqrt_MVP(V):
            delta_hdiag_v = einsum("ia,iam->iam", delta_hdiag_sqrt, V)
            return delta_hdiag_v
        return hdiag_sqrt_MVP, hdiag_sq

def gen_delta_hdiag_MVP(delta_hdiag):
    def hdiag_MVP(V):
        delta_hdiag_v = einsum("ia,iam->iam", delta_hdiag, V)
        return delta_hdiag_v
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
        '''
        T_right_jb_V = einsum("jbP,jbm->Pm", T_right, V)
        iajb_V = einsum("iaP,Pm->iam", T_left, T_right_jb_V)
        # print('iajb_V.dtype', iajb_V.dtype)
        return iajb_V
    return iajb_MVP

def gen_ijab_MVP(T_ij, T_ab):
    def ijab_MVP(V):
        '''
        (ij|ab) = Σ_Pjb (T_ij^P T_ab^P V_jb^m)
                = Σ_P [T_ij^P Σ_jb(T_ab^P V_jb^m)]
        '''
        T_ab_V = einsum("abP,jbm->jPam", T_ab, V)
        ijab_V = einsum("ijP,jPam->iam", T_ij, T_ab_V)
        # print('ijab_V.dtype', ijab_V.dtype)
        return ijab_V
    return ijab_MVP

def get_ibja_MVP(T_ia):    
    def ibja_MVP(V):
        '''
        the exchange (ib|ja) in B matrix
        (ib|ja) = Σ_Pjb (T_ib^P T_ja^P V_jb^m)
                = Σ_P [T_ja^P Σ_jb(T_ib^P V_jb^m)]           
        '''
        T_ib_V = einsum("ibP,jbm->Pijm", T_ia, V)
        ibja_V = einsum("jaP,Pijm->iam", T_ia, T_ib_V)
        # print('ibja_V.dtype',ibja_V.dtype)
        return ibja_V
    return ibja_MVP


class TDDFT_ris(object):
    def __init__(self, 
                mf: dft, 
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
                single: bool = False):

        self.mf = mf
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
        self.single = single
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

        if self.mol.cart:
            self.eri_tag = '_cart'
        else:
            self.eri_tag = '_sph'
        print('cartesian or spherical electron integral =',self.eri_tag)


        if single == True:
            mf.mo_coeff = mf.mo_coeff.astype(np.float32)

        if mf.mo_coeff.ndim == 2:
            self.RKS = True
            self.UKS = False
            # self.n_if = len(mf.mo_occ)
            n_occ = sum(mf.mo_occ>0)
            n_vir = sum(mf.mo_occ==0)           
            self.n_occ = n_occ
            self.n_vir = n_vir

            self.C_occ_notrunc = mf.mo_coeff[:,:n_occ]
            self.C_vir_notrunc = mf.mo_coeff[:,n_occ:]

            mo_energy = mf.mo_energy

            vir_ene = mo_energy[n_occ:].reshape(1,n_vir)
            occ_ene = mo_energy[:n_occ].reshape(n_occ,1)
     
            delta_hdiag = np.repeat(vir_ene, n_occ, axis=0) - np.repeat(occ_ene, n_vir, axis=1)
            self.delta_hdiag = delta_hdiag

            print('n_occ =', n_occ)
            print('n_vir =', n_vir)

            if Ktrunc > 0:
                print(f' MO truncation in K with threshold {Ktrunc} eV above HOMO and below LUMO')

                trunc_tol_au = Ktrunc/parameter.Hartree_to_eV     

                homo_vir_delta_ene = delta_hdiag[-1,:]
                occ_lumo_delta_ene = delta_hdiag[:,0]
                
                rest_occ = np.sum(occ_lumo_delta_ene <= trunc_tol_au)
                rest_vir = np.sum(homo_vir_delta_ene <= trunc_tol_au)
                
            elif Ktrunc == 0: 
                print('no MO truncation in K')
                rest_occ = n_occ
                rest_vir = n_vir 
                
            print('rest_occ =',rest_occ)
            print('rest_vir =',rest_vir)

            self.C_occ_Ktrunc = mf.mo_coeff[:,n_occ-rest_occ:n_occ]
            self.C_vir_Ktrunc = mf.mo_coeff[:,n_occ:n_occ+rest_vir]
             
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

        delta_hdiag = self.delta_hdiag

        mol = self.mol
        theta = self.theta 

        J_fit = self.J_fit
        K_fit = self.K_fit

        alpha = self.alpha
        beta = self.beta
        omega = self.omega

        ''' RIJ '''
        auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)

        eri2c_J, eri3c_J = get_eri2c_eri3c(mol=mol, auxmol=auxmol_J, omega=0, single=single)
        uvP_withL_J = get_uvP_withL(eri2c=eri2c_J, eri3c=eri3c_J)


        ''' RIK '''
        if K_fit == J_fit:
            auxmol_K = auxmol_J
            eri2c_K = eri2c_J
            eri3c_K = eri3c_J
            if omega == None or omega == 0:
                ''' just normal hybrid, go ahead to build uvP_withL_K '''
                uvP_withL_K = uvP_withL_J
        else:
            auxmol_K = get_auxmol(mol=mol, theta=theta, fitting_basis=K_fit) 
            eri2c_K, eri3c_K = get_eri2c_eri3c(mol=mol, auxmol=auxmol_K, omega=0)
            if omega == None or omega == 0:
                ''' just normal hybrid, go ahead to build uvP_withL_K '''
                uvP_withL_K = get_uvP_withL(eri2c=eri2c_K, eri3c=eri3c_K)

        if omega and omega > 0:
            '''
            the 2c2e and 3c2e integrals for hybrid or RSH (range-saparated hybrid) 

            (ij|ab) = (ij|1-(alpha + beta*erf(omega))/r|ab)  + (ij|alpha + beta*erf(omega)/r|ab)
            short-range part (ij|1-(alpha + beta*erf(omega))/r|ab) is treated by the DFT XC functional, thus not considered here
            long-range part  (ij|alpha + beta*erf(omega)/r|ab) = alpha (ij|r|ab) + beta*(ij|erf(omega)/r|ab)
            ''' 

            print(f'rebuild eri2c and eri3c with screening factor ω = {omega}')
            '''RSH, eri2c_K and eri3c_K need to be redefined'''
            eri2c_K, eri3c_K = get_eri2c_eri3c_RSH(mol=mol, 
                                                auxmol=auxmol_K, 
                                                eri2c_K=eri2c_K, 
                                                eri3c_K=eri3c_K, 
                                                alpha=alpha, 
                                                beta=beta, 
                                                omega=omega, 
                                                single=single)

            uvP_withL_K = get_uvP_withL(eri2c=eri2c_K, eri3c=eri3c_K)


        hdiag = delta_hdiag.reshape(-1)
        delta_hdiag_MVP = gen_delta_hdiag_MVP(delta_hdiag)
        '''hybrid RKS TDA'''
        
        T_ia_J = get_Tia(uvP_withL=uvP_withL_J, C_occ=C_occ_notrunc, C_vir=C_vir_notrunc)
        T_ij_K, T_ab_K = get_Tij_Tab(uvP_withL=uvP_withL_K, C_occ=C_occ_Ktrunc, C_vir=C_vir_Ktrunc)


        iajb_MVP = gen_iajb_MVP(T_left=T_ia_J, T_right=T_ia_J)
        ijab_MVP = gen_ijab_MVP(T_ij=T_ij_K,   T_ab=T_ab_K)
        
        def RKS_TDA_hybrid_MVP(X):
            ''' hybrid or range-sparated hybrid, a_x > 0
                return AX
                AV = delta_hdiag_MVP(V) + 2*iajb_MVP(V) - a_x*ijab_MVP(V)
                for RSH, a_x = 1

                if not MO truncation, then n_occ-rest_occ=0 and rest_vir=n_vir
            '''
            # print('a_x=', a_x)
            X = X.reshape(n_occ, n_vir, -1)
            AX = delta_hdiag_MVP(X) 
            AX += 2 * iajb_MVP(X) 
            AX[n_occ-rest_occ:,:rest_vir,:] -= a_x * ijab_MVP(X[n_occ-rest_occ:,:rest_vir,:])
            AX = AX.reshape(n_occ*n_vir, -1)

            return AX

        return RKS_TDA_hybrid_MVP, hdiag
            

    def gen_RKS_TDDFT_hybrid_MVP(self):  
        '''hybrid RKS TDA'''
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

        delta_hdiag = self.delta_hdiag

        mol = self.mol
        theta = self.theta 

        J_fit = self.J_fit
        K_fit = self.K_fit

        alpha = self.alpha
        beta = self.beta
        omega = self.omega


        ''' RIJ '''
        auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)
        
        eri2c_J, eri3c_J = get_eri2c_eri3c(mol=mol, auxmol=auxmol_J, omega=0, single=single)
        print_memory_usage('RIJ eri3c')
        # uvP_withL_J = get_uvP_withL(eri2c=eri2c_J, eri3c=eri3c_J)
        # print_memory_usage('RIJ uvP_withL_J')

        ''' RIK '''
        if K_fit == J_fit:
            ''' K uese exactly same basis as J and they share same set of Tensors'''
            auxmol_K = auxmol_J
            eri2c_K, eri3c_K = eri2c_J, eri3c_J

            # if omega == None or omega == 0:
            #     ''' just normal hybrid, go ahead to build uvP_withL_K '''
            #     uvP_withL_K = uvP_withL_J

        else:
            ''' K uese different basis as J'''
            auxmol_K = get_auxmol(mol=mol, theta=theta, fitting_basis=K_fit) 
            eri2c_K, eri3c_K = get_eri2c_eri3c(mol=mol, auxmol=auxmol_K, omega=0, single=single)
            # if omega == None or omega == 0:
            #     ''' just normal hybrid, go ahead to build uvP_withL_K '''
            #     uvP_withL_K = get_uvP_withL(eri2c=eri2c_K, eri3c=eri3c_K)

        if omega and omega > 0:
            print(f'rebuild eri2c_K and eri3c_K with screening factor ω = {omega}')
            '''RSH, eri2c_K and eri3c_K need to be redefined'''
            eri2c_K, eri3c_K = get_eri2c_eri3c_RSH(mol=mol, 
                                                auxmol=auxmol_K, 
                                                eri2c_K=eri2c_K, 
                                                eri3c_K=eri3c_K, 
                                                alpha=alpha, 
                                                beta=beta, 
                                                omega=omega, 
                                                single=single)

            # uvP_withL_K = get_uvP_withL(eri2c=eri2c_K, eri3c=eri3c_K)

        hdiag = delta_hdiag.reshape(-1)
        delta_hdiag_MVP = gen_delta_hdiag_MVP(delta_hdiag)
        
        
        # T_ia_J = get_Tia(uvP_withL=uvP_withL_J, C_occ=C_occ_notrunc, C_vir=C_vir_notrunc)
        lower_inv_eri2c_J = np.linalg.cholesky(np.linalg.inv(eri2c_J))
        T_ia_J = get_Tia(eri3c=eri3c_J, lower_inv_eri2c=lower_inv_eri2c_J, C_occ=C_occ_notrunc, C_vir=C_vir_notrunc)

        # T_ia_K = get_Tia(uvP_withL=uvP_withL_K, C_occ=C_occ_Ktrunc, C_vir=C_vir_Ktrunc)
        lower_inv_eri2c_K = np.linalg.cholesky(np.linalg.inv(eri2c_K))
        T_ia_K = get_Tia(eri3c=eri3c_K, lower_inv_eri2c=lower_inv_eri2c_K, C_occ=C_occ_Ktrunc, C_vir=C_vir_Ktrunc)

        T_ij_K, T_ab_K = get_Tij_Tab(eri3c=eri3c_K, lower_inv_eri2c=lower_inv_eri2c_K, C_occ=C_occ_Ktrunc, C_vir=C_vir_Ktrunc)


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

            (A+B)V = delta_hdiag_MVP(V) + 4*iajb_MVP(V) - a_x * [ ijab_MVP(V) + ibja_MVP(V) ]
            (A-B)V = delta_hdiag_MVP(V) - a_x * [ ijab_MVP(V) - ibja_MVP(V) ]
            for RSH, a_x = 1, because the exchange component is defined by alpha+beta (alpha+beta not awlways == 1)
            '''
            X = X.reshape(n_occ, n_vir, -1)
            Y = Y.reshape(n_occ, n_vir, -1)

            XpY = X + Y
            XmY = X - Y

            ApB_XpY = delta_hdiag_MVP(XpY) 
            ApB_XpY += 4*iajb_MVP(XpY) 
            ApB_XpY[n_occ-rest_occ:,:rest_vir,:] -= a_x*ijab_MVP(XpY[n_occ-rest_occ:,:rest_vir,:]) 
            ApB_XpY[n_occ-rest_occ:,:rest_vir,:] -= a_x*ibja_MVP(XpY[n_occ-rest_occ:,:rest_vir,:])

            AmB_XmY = delta_hdiag_MVP(XmY) 
            AmB_XmY[n_occ-rest_occ:,:rest_vir,:] -= a_x*ijab_MVP(XmY[n_occ-rest_occ:,:rest_vir,:]) 
            AmB_XmY[n_occ-rest_occ:,:rest_vir,:] += a_x*ibja_MVP(XmY[n_occ-rest_occ:,:rest_vir,:])

            ''' (A+B)(X+Y) = AX + BY + AY + BX   (1)
                (A-B)(X-Y) = AX + BY - AY - BX   (2)
                (1) + (1) /2 = AX + BY = U1
                (1) - (2) /2 = AY + BX = U2
            '''
            U1 = (ApB_XpY + AmB_XmY)/2
            U2 = (ApB_XpY - AmB_XmY)/2

            U1 = U1.reshape(n_occ*n_vir,-1)
            U2 = U2.reshape(n_occ*n_vir,-1)

            return U1, U2
        return RKS_TDDFT_hybrid_MVP, hdiag


    #  ===========  RKS pure ===========
    def get_RKS_TDA_pure_MVP(self):
        a_x = self.a_x
        n_occ = self.n_occ
        n_vir = self.n_vir
        # n_occ*n_vir = n_occ*n_vir

        mo_coeff = self.mf.mo_coeff
        mo_energy = self.mf.mo_energy

        mol = self.mol
        auxmol = self.get_auxmol(theta=self.theta, add_p=self.add_p)
        eri2c, eri3c = self.get_eri2c_eri3c(mol=self.mol, auxmol=auxmol, omega=0)

        uvP_withL = get_uvP_withL(eri2c=eri2c, eri3c=eri3c)

        raise ValueError('not implemented yet')
        '''pure RKS TDA'''
        T_ia = get_Tia(uvP_withL, mo_coeff, n_occ) 

        iajb_MVP = self.gen_iajb_MVP(T_left=T_ia, T_right=T_ia)
        def RKS_TDA_pure_MVP(X):
            ''' pure functional, a_x = 0
                return AX
                AV = hdiag_MVP(V) + 2*iajb_MVP(V) 
                for RSH, a_x = 1
            '''
            # print('a_x=', a_x)
            X = X.reshape(n_occ, n_vir, -1)
            AX = hdiag_MVP(X) + 2*iajb_MVP(X)
            AX = AX.reshape(n_occ*n_vir, -1)
            return AX

        return RKS_TDA_pure_MVP, hdiag
       
    def gen_RKS_TDDFT_pure_MVP(self):  
            
            a_x = self.a_x
            n_occ = self.n_occ
            n_vir = self.n_vir
            # n_occ*n_vir=n_occ*n_vir 

            mo_coeff = self.mf.mo_coeff
            mo_energy = self.mf.mo_energy

            mol = self.mol
            auxmol = self.get_auxmol(theta=self.theta, add_p=self.add_p)
            eri2c, eri3c = self.get_eri2c_eri3c(mol=self.mol, auxmol=auxmol, omega=0)
            uvP_withL = self.get_uvP_withL(eri2c=eri2c, eri3c=eri3c)

            hdiag_MVP, hdiag = self.get_hdiag_MVP(mo_energy=mo_energy, 
                                                    n_occ=n_occ, 
                                                    n_vir=n_vir)
            '''pure RKS TDDFT'''
            hdiag_sqrt_MVP, hdiag_sq = self.get_hdiag_MVP(mo_energy=mo_energy, 
                                                          n_occ=n_occ, 
                                                          n_vir=n_vir,
                                                          sqrt=True)
            T_ia = self.get_T(uvP_withL=uvP_withL,
                            n_occ=n_occ, 
                            mo_coeff=mo_coeff,
                            calc='coulomb_only')
            iajb_MVP = self.gen_iajb_MVP(T_left=T_ia, T_right=T_ia)

            def RKS_TDDFT_pure_MVP(Z):
                '''(A-B)^1/2(A+B)(A-B)^1/2 Z = Z w^2
                                        MZ = Z w^2
                    X+Y = (A-B)^1/2 Z
                    A+B = hdiag_MVP(V) + 4*iajb_MVP(V)
                    (A-B)^1/2 = hdiag_sqrt_MVP(V)
                '''
                Z = Z.reshape(n_occ, n_vir, -1)
                AmB_sqrt_V = hdiag_sqrt_MVP(Z)
                ApB_AmB_sqrt_V = hdiag_MVP(AmB_sqrt_V) + 4*iajb_MVP(AmB_sqrt_V)
                MZ = hdiag_sqrt_MVP(ApB_AmB_sqrt_V)
                MZ = MZ.reshape(n_occ*n_vir, -1)
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
        hdiag = np.vstack((hdiag_a.reshape(-1,1), hdiag_b.reshape(-1,1))).reshape(-1)

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

                U = np.vstack((U_a, U_b))
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

                U = np.vstack((U_a, U_b))
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
        hdiag = np.vstack((hdiag_a.reshape(-1,1), hdiag_b.reshape(-1,1))).reshape(-1)

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
                ApB_XpY = np.vstack((ApB_XpY_alpha, ApB_XpY_beta))

                AmB_XmY_alpha = AmB_XmY_aa.reshape(A_aa_size,-1)
                AmB_XmY_beta  = AmB_XmY_bb.reshape(A_bb_size,-1)
                AmB_XmY = np.vstack((AmB_XmY_alpha, AmB_XmY_beta))
                
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
            hdiag_sq = np.vstack((hdiag_a_sq.reshape(-1,1), hdiag_b_sq.reshape(-1,1))).reshape(-1)

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

                MZ = np.vstack((MZ_a, MZ_b))
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
        
    # def get_vind(self, TDA_MVP, TDDFT_MVP):
    #     '''
    #     _vind is pyscf style, to feed pyscf eigensovler
    #     _MVP is my style, to feed my eigensovler
    #     '''
    #     def TDA_vind(V):
    #         '''
    #         return AX
    #         '''
    #         V = np.asarray(V)
    #         return TDA_MVP(V.T).T

    #     def TDDFT_vind(U):
    #         # print('U.shape',U.shape)
    #         X = U[:,:n_occ*n_vir].T
    #         Y = U[:,n_occ*n_vir:].T
    #         U1, U2 = TDDFT_MVP(X, Y)
    #         U = np.vstack((U1, U2)).T
    #         return U

    #     return TDA_vind, TDDFT_vind

    def get_P(self, int_r, mo_coeff, mo_occ):
        '''
        transition dipole
        mol: mol obj
        '''
        occidx = mo_occ > 0
        orbo = mo_coeff[:, occidx]
        orbv = mo_coeff[:,~occidx]

        P = einsum("xpq,pi,qa->iax", int_r, orbo, orbv.conj())
        P = P.reshape(-1,3)
        return P

    def get_RKS_P(self):
        '''
        transition dipole P
        '''
        tag = self.eri_tag
        int_r = self.mol.intor_symmetric('int1e_r'+tag)
        mo_occ = self.mf.mo_occ
        mo_coeff = self.mf.mo_coeff

        P = self.get_P(int_r=int_r, mo_coeff=mo_coeff, mo_occ=mo_occ)
        P = P.reshape(-1,3)
        return P

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

        P_alpha = self.get_P(int_r=int_r, mo_coeff=mo_coeff[0], mo_occ=mo_occ[0])
        P_beta = self.get_P(int_r=int_r, mo_coeff=mo_coeff[1], mo_occ=mo_occ[1])
        P = np.vstack((P_alpha, P_beta))
        return P

    def kernel_TDA(self):
 
        '''for TDA, pure and hybrid share the same form of
                     AX = Xw
            always use the Davidson solver
            pure TDA is not using MZ=Zw^2 form
        '''

        if self.RKS:
            if self.a_x != 0:
                TDA_MVP, hdiag = self.get_RKS_TDA_hybrid_MVP()
                P = self.get_RKS_P()
            elif self.a_x == 0:
                TDA_MVP, hdiag = self.get_RKS_TDA_pure_MVP()
                P = self.get_RKS_P()

        elif self.UKS:
            TDA_MVP, hdiag = self.get_UKS_TDA_MVP()
            P = self.get_UKS_P()


        # print('min(hdiag)', min(hdiag)*parameter.Hartree_to_eV)
        energies, X = eigen_solver.Davidson(matrix_vector_product=TDA_MVP,
                                            hdiag=hdiag,
                                            N_states=self.nroots,
                                            conv_tol=self.conv_tol,
                                            max_iter=self.max_iter,
                                            single=self.single)
        energies = energies*parameter.Hartree_to_eV
        # print('energies =', energies)

        # print('self.print_threshold', self.print_threshold)
        oscillator_strength = spectralib.get_spectra(energies=energies, 
                                                       transition_vector= X, 
                                                       X=X/(2**0.5),
                                                       Y=None,
                                                       P=P, 
                                                       name=self.out_name+'_TDA_ris', 
                                                       RKS=self.RKS,
                                                       spectra=self.spectra,
                                                       print_threshold = self.print_threshold,
                                                       n_occ=self.n_occ if self.RKS else (self.n_occ_a, self.n_occ_b),
                                                       n_vir=self.n_vir if self.RKS else (self.n_vir_a, self.n_vir_b))

        return energies, X, oscillator_strength
        # from pyscf.lib import davidson1
        # TDA_vind, _  = self.get_vind()

        # def TDA_diag_initial_guess(N_states, hdiag):
        #     '''
        #     m is the amount of initial guesses
        #     '''
        #     hdiag = hdiag.reshape(-1,)
        #     V_size = hdiag.shape[0]
        #     Dsort = hdiag.argsort()
        #     energies = hdiag[Dsort][:N_states]
        #     V = np.zeros((V_size, N_states))
        #     for j in range(N_states):
        #         V[Dsort[j], j] = 1.0
        #     return V
        
        # initial = TDA_diag_initial_guess(self.nroots, hdiag).T
        # converged, e, amps = davidson1(
        #               aop=TDA_vind, x0=initial, precond=hdiag,
        #               tol=self.conv_tol,
        #               nroots=self.nroots, lindep=1e-14,
        #               max_cycle=35,
        #               max_space=1000)
        # e*=parameter.Hartree_to_eV
        # return converged, e, amps

    # def kernel_TDDFT(self):
        # '''
        # use pyscf davidson solver to solve TDDFT
        # but pyscf davidson solver is not relibale, so we use our own davidson solver
        # '''
    #     TDA_MVP, TDDFT_MVP, TDA_vind, TDDFT_vind, hdiag, Hdiag = self.get_vind()
        #  def TDDFT_diag_initial_guess(N_states, hdiag):
        #     '''
        #     m is the amount of initial guesses
        #     '''
        #     hdiag = hdiag.reshape(-1,)
        #     V_size = hdiag.shape[0]
        #     Dsort = hdiag.argsort()
        #     energies = hdiag[Dsort][:N_states]
        #     print('energies =', energies)
        #     V = np.zeros((V_size, N_states))
        #     for j in range(N_states):
        #         V[Dsort[j], j] = 1.0
        #     V2 = np.zeros_like(V)
        #     V_long = np.vstack((V, V2))
        #     return V_long
    #     initial = TDDFT_diag_initial_guess(self.nroots, hdiag).T
    #     print('initial.shape',initial.shape)
    #     converged, e, amps = davidson_nosym1(
    #                   aop=TDDFT_vind, x0=initial, precond=Hdiag,
    #                   tol=self.conv_tol,
    #                   nroots=self.nroots, lindep=1e-14,
    #                   max_cycle=35,
    #                   max_space=10000)
    #     e*=parameter.Hartree_to_eV
    #     return converged, e, amps


        # if self.pyscf_TDDFT_vind:
        #     '''
        #     invoke ab-initio TDDFT from PySCF and use our davidson solver
        #     '''
        #     name = 'TDDFT-abinitio'
        #     def TDDFT_MVP(X, Y):
        #         '''convert pyscf style (bra) to my style (ket)
        #         return AX + BY and AY + BX'''
        #         XY = np.vstack((X,Y)).T
        #         U = self.pyscf_TDDFT_vind(XY)
        #         n_occ*n_vir = U.shape[1]//2
        #         U1 = U[:,:n_occ*n_vir].T
        #         U2 = -U[:,n_occ*n_vir:].T
        #         return U1, U2
        # else:
        #     name = 'TDDFT-ris'
            
    def kernel_TDDFT(self):     
        # math_helper.show_memory_info('At the beginning')
        if self.a_x != 0:
            '''hybrid TDDFT'''
            if self.RKS:
                P = self.get_RKS_P()
                TDDFT_hybrid_MVP, hdiag = self.gen_RKS_TDDFT_hybrid_MVP()

            elif self.UKS:
                P = self.get_UKS_P()
                TDDFT_hybrid_MVP, hdiag = self.get_UKS_TDDFT_MVP()
            # math_helper.show_memory_info('After get_TDDFT_MVP')
            energies, X, Y = eigen_solver.Davidson_Casida(TDDFT_hybrid_MVP, hdiag,
                                                            N_states = self.nroots,
                                                            conv_tol = self.conv_tol,
                                                            max_iter = self.max_iter,
                                                            single=self.single)
        elif self.a_x == 0:
            '''pure TDDFT'''
            if self.RKS:
                TDDFT_pure_MVP, hdiag_sq = self.get_RKS_TDDFT_MVP()
                P = self.get_RKS_P()

            elif self.UKS:
                TDDFT_pure_MVP, hdiag_sq = self.get_UKS_TDDFT_MVP()
                P = self.get_UKS_P()
            # print('min(hdiag_sq)', min(hdiag_sq)**0.5*parameter.Hartree_to_eV)
            energies_sq, Z = eigen_solver.Davidson(TDDFT_pure_MVP, hdiag_sq,
                                                N_states=self.nroots,
                                                conv_tol=self.conv_tol,
                                                max_iter=self.max_iter,
                                                single=self.single)
            
            # print('check norm of Z', np.linalg.norm(np.dot(Z.T,Z) - np.eye(Z.shape[1])))
            energies = energies_sq**0.5
            Z = Z*energies**0.5

            X, Y = math_helper.XmY_2_XY(Z=Z, AmB_sq=hdiag_sq, omega=energies)

        XY_norm_check = np.linalg.norm( (np.dot(X.T,X) - np.dot(Y.T,Y)) -np.eye(self.nroots) )
        print('check norm of X^TX - Y^YY - I = {:.2e}'.format(XY_norm_check))

        energies = energies*parameter.Hartree_to_eV

        oscillator_strength = spectralib.get_spectra(energies=energies, 
                                                    transition_vector= X+Y, 
                                                    X = X/(2**0.5),
                                                    Y = Y/(2**0.5),
                                                    P=P, 
                                                    name=self.out_name+'_TDDFT_ris', 
                                                    spectra=self.spectra,
                                                    RKS=self.RKS,
                                                    print_threshold = self.print_threshold,
                                                    n_occ=self.n_occ if self.RKS else (self.n_occ_a, self.n_occ_b),
                                                    n_vir=self.n_vir if self.RKS else (self.n_vir_a, self.n_vir_b))
        
        # print('energies =', energies)
        return energies, X, Y, oscillator_strength

