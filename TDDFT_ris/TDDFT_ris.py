from pyscf import gto, lib
from pyscf.lib import davidson1, davidson_nosym1
from collections import Counter
import numpy as np
import os,sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)


from TDDFT_ris import parameter, math_helper, eigen_solver, diag_ip

einsum = lib.einsum

from pyscf.lib import davidson1, davidson_nosym1

def TDA_diag_initial_guess(N_states, hdiag):
    '''
    m is the amount of initial guesses
    '''
    hdiag = hdiag.reshape(-1,)
    V_size = hdiag.shape[0]
    Dsort = hdiag.argsort()
    energies = hdiag[Dsort][:N_states]
    V = np.zeros((V_size, N_states))
    for j in range(N_states):
        V[Dsort[j], j] = 1.0
    return V

def TDDFT_diag_initial_guess(N_states, hdiag):
    '''
    m is the amount of initial guesses
    '''
    hdiag = hdiag.reshape(-1,)
    V_size = hdiag.shape[0]
    Dsort = hdiag.argsort()
    energies = hdiag[Dsort][:N_states]
    V = np.zeros((V_size, N_states))
    for j in range(N_states):
        V[Dsort[j], j] = 1.0
    V2 = np.zeros_like(V)
    V_long = np.vstack((V, V2))
    return V_long

def copy_array(A):
    B = np.zeros_like(A)
    dim = len(B.shape)
    if dim == 1:
        B[:,] = A[:,]
    elif dim == 2:
        B[:,:] = A[:,:]
    elif dim == 3:
        B[:,:,:] = A[:,:,:]
    return B

def gen_P(mf, mol):
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:,~occidx]
    int_r= mol.intor_symmetric('int1e_r')
    P = lib.einsum("xpq,pi,qa->iax", int_r, orbo, orbv.conj())
    P = P.reshape(-1,3)
    return P

class TDDFT_ris(object):
    def __init__(self, mf, mol, theta=0.2,
                                add_p=False,
                             conv_tol=1e-10,
                               nroots=5,
                             max_iter=25,
                    pyscf_TDDFT_vind = None):
        '''add_p: whether add p orbital to aux basis
        '''
        self.mf = mf
        self.mol = mol
        self.add_p = add_p
        self.theta = theta
        self.conv_tol = conv_tol
        self.nroots = nroots
        self.max_iter = max_iter
        self.pyscf_TDDFT_vind = pyscf_TDDFT_vind

    def gen_auxmol(self, U=0.2, add_p=False, full_fitting=False):
        print('asigning auxiliary basis set, add p function =', add_p)
        print('U =', U)
        '''
        parse_arg = False turns off PySCF built-in output file
        '''
        auxmol = gto.M(atom=self.mol.atom, parse_arg = False)

        if not full_fitting:
            atom_count = Counter(auxmol.elements)

            '''
            auxmol_basis_keys = ['C', 'H', 'H^2', 'H^3', 'H^4', 'O'}
            '''
            auxmol_basis_keys = []
            for key in atom_count:
                for i in range(atom_count[key]):
                    if i > 0:
                        auxmol_basis_keys.append(key+'^'+str(i+1))
                    else:
                        auxmol_basis_keys.append(key)
            # print('auxmol_basis_keys', auxmol_basis_keys)
            '''
            aux_basis = {
            'C': [[0, [0.123, 1.0]], [1, [0.123, 1.0]]],
            'H': [[0, [0.123, 1.0]]],
            'H^2': [[0, [0.123, 1.0]]],
            'H^3': [[0, [0.123, 1.0]]],
            'H^4': [[0, [0.123, 1.0]]],
            'O': [[0, [0.123, 1.0]], [1, [0.123, 1.0]]]
            }
            '''
            aux_basis = {}
            for i in range(len(auxmol_basis_keys)):
                atom_index = auxmol_basis_keys[i]
                atom = atom_index.split('^')[0]

                exp = parameter.as_exp[atom] * U

                if atom != 'H' and add_p == True:
                    aux_basis[atom_index] = [[0, [exp, 1.0]],[1, [exp, 1.0]]]
                else:
                    aux_basis[atom_index] = [[0, [exp, 1.0]]]

        else:
            print('full aux_basis')
            aux_basis = args.basis_set+"-jkfit"
        auxmol.basis = aux_basis
        auxmol.build()
        # print(auxmol._basis)
        # [print(k, v) for k, v in auxmol.basis.items()]

        return auxmol


    def gen_2c_3c(self, mol, auxmol, RSH_omega=0):

        nao = mol.nao_nr()
        naux = auxmol.nao_nr()

        mol.set_range_coulomb(RSH_omega)
        auxmol.set_range_coulomb(RSH_omega)

        '''
        (pq|rs) = Σ_PQ (pq|P)(P|Q)^-1(Q|rs)
        2 center 2 electron integral (P|Q)
        N_auxbf * N_auxbf
        '''
        eri2c = auxmol.intor('int2c2e_sph')
        '''
        3 center 2 electron integral (pq|P)
        N_bf * N_bf * N_auxbf
        '''
        pmol = mol + auxmol
        eri3c = pmol.intor('int3c2e_sph',
                            shls_slice=(0,mol.nbas,0,mol.nbas,
                            mol.nbas,mol.nbas+auxmol.nbas))
        return eri2c, eri3c

    def gen_electron_int(self, mol, auxmol_cl, auxmol_ex, alpha_RSH=None, beta_RSH=None):

        eri2c_cl, eri3c_cl = self.gen_2c_3c(mol=mol, auxmol=auxmol_cl, RSH_omega=0)
        eri2c_ex, eri3c_ex = self.gen_2c_3c(mol=mol, auxmol=auxmol_ex, RSH_omega=0)

        if alpha_RSH:
            print('2c2e and 2c2e for RSH (ij|ab)')
            eri2c_erf, eri3c_erf = self.gen_2c_3c(mol=mol, auxmol=auxmol_ex, RSH_omega=0.3)
            eri2c_ex = alpha_RSH*eri2c_ex + beta_RSH*eri2c_erf
            eri3c_ex = alpha_RSH*eri3c_ex + beta_RSH*eri3c_erf
        else:
            print('2c2e and 3c2e for (ia|jb)')

        return eri2c_cl, eri3c_cl, eri2c_ex, eri3c_ex


    def gen_GAMMA(self, eri2c, eri3c, calc, n_occ, un_ortho_C_matrix):

        N_auxbf = eri2c.shape[0]

        '''
        PQ is eri2c shape, N_auxbf
        GAMMA.shape = (N_bf, N_bf, N_auxbf)
        '''
        Delta = einsum("PQ,uvQ->uvP", np.linalg.inv(eri2c), eri3c)
        GAMMA = einsum("up,vq,uvP->pqP", un_ortho_C_matrix, un_ortho_C_matrix, Delta)

        '''
        N_bf:
      ==============|---------------------------------------------
            n_occ                         n_vir

                     n_occ          n_vir
                -|-------------||-------------|
                 |             ||             |
           n_occ |   GAMMA_ij  ||  GAMMA_ia   |
                 |             ||             |
                 |             ||             |
                =|=============||=============|
                 |             ||             |
           n_vir |             ||  GAMMA_ab   |
                 |             ||             |
                 |             ||             |
                -|-------------||-------------|

        '''

        GAMMA_ia = copy_array(GAMMA[:n_occ,n_occ:,:])
        GAMMA_ia_B = einsum("iaA,AB->iaB", GAMMA_ia, eri2c)

        if calc == 'coulomb':
            '''(ia|jb) coulomb term'''

            diag_cl = einsum("iaA,iaA->ia", GAMMA_ia_B, GAMMA_ia)

            return GAMMA_ia, GAMMA_ia_B

        if calc == 'exchange':
            '''(ij|ab) exchange term '''

            GAMMA_ij = copy_array(GAMMA[:n_occ, :n_occ,:])
            GAMMA_ab = copy_array(GAMMA[n_occ:,n_occ:,:])
            GAMMA_ij_B = einsum("ijA,AB->ijB", GAMMA_ij, eri2c)

            return GAMMA_ia, GAMMA_ia_B, GAMMA_ab, GAMMA_ij_B

    '''
    use coulomb type integral without RSH,
    only the exchange type integral with RSH
    '''
    def gen_coulomb(self, GAMMA_ia, GAMMA_ia_B):
        def iajb_fly(V):
            '''(ia|jb)'''
            GAMMA_jb_V = einsum("iaA,iam->Am", GAMMA_ia, V)
            iajb_V = einsum("iaA,Am->iam", GAMMA_ia_B, GAMMA_jb_V)
            return iajb_V
        return iajb_fly

    def gen_exchange(self, GAMMA_ia, GAMMA_ia_B, GAMMA_ab, GAMMA_ij_B):
        def ijab_fly(V):
            '''(ij|ab)'''
            GAMMA_ab_V = einsum("abA,jbm->jAam", GAMMA_ab, V)
            ijab_V  = einsum("ijA,jAam->iam", GAMMA_ij_B, GAMMA_ab_V)
            return ijab_V

        def ibja_fly(V):
            '''
            the Forck exchange energy in B matrix
            (ib|ja)
            '''
            GAMMA_ja_V = einsum("ibA,jbm->Ajim", GAMMA_ia, V)
            ibja_V = einsum("jaA,Ajim->iam", GAMMA_ia_B, GAMMA_ja_V)
            return ibja_V

        return ijab_fly, ibja_fly

    def gen_delta_hdiag_fly(self, delta_hdiag):
        def delta_hdiag_fly(V):
            '''
            delta_hdiag.shape = (n_occ, n_vir)
            '''
            delta_hdiag_v = einsum("ia,iam->iam", delta_hdiag, V)
            return delta_hdiag_v
        return delta_hdiag_fly

    def gen_mv_fly(self, delta_hdiag_fly, iajb_fly, ijab_fly, ibja_fly, a_x,
                                         n_occ, n_vir, A_size):

        def TDA_mv(X):
            ''' return AX
                for RSH, a_x = 1
                AV = delta_fly(V) + 2*iajb_fly(V) - a_x*ijab_fly(V)
            '''
            # print('a_x=', a_x)
            X = X.reshape(n_occ, n_vir, -1)

            AX = delta_hdiag_fly(X) + 2*iajb_fly(X) - a_x*ijab_fly(X)
            AX = AX.reshape(A_size, -1)

            return AX

        def TDDFT_mv(X, Y):
            '''
            [A B][X] = [AX+BY] = [U1]
            [B A][Y]   [AY+BX]   [U2]
            we want AX+BY and AY+BX
            instead of computing AX+BY and AY+BX directly
            we compute (A+B)(X+Y) and (A-B)(X-Y)
            it can save one tensor contraction compared to directly computing AX+BY and AY+BX

            (A+B)V = delta_fly(V) + 4*iajb_fly(V) - a_x*[ibja_fly(V) + ijab_fly(V)]
            (A-B)V = delta_fly(V) + a_x[ibja_fly(V) - ijab_fly(V)]
            (for RSH, a_x = 1)
            '''
            X = X.reshape(n_occ, n_vir, -1)
            Y = Y.reshape(n_occ, n_vir, -1)

            X_p_Y = X + Y
            X_m_Y = X - Y

            A_p_B_X_p_Y = delta_hdiag_fly(X_p_Y) + 4*iajb_fly(X_p_Y) - a_x*(ibja_fly(X_p_Y) + ijab_fly(X_p_Y))

            A_m_B_X_m_Y = delta_hdiag_fly(X_m_Y) + a_x*(ibja_fly(X_m_Y) - ijab_fly(X_m_Y))

            U1 = (A_p_B_X_p_Y + A_m_B_X_m_Y)/2
            U2 = (A_p_B_X_p_Y - A_m_B_X_m_Y)/2

            U1 = U1.reshape(A_size,-1)
            U2 = U2.reshape(A_size,-1)

            return U1, U2

        def TDDFT_spolar_mv(X):

            ''' for RSH, a_x=1
                (A+B)X = delta_fly(V) + 4*iajb_fly(V) - a_x*[ijab_fly(V) + ibja_fly(V)]
            '''
            X = X.reshape(n_occ, n_vir, -1)

            ABX = delta_hdiag_fly(X) + 4*iajb_fly(X) - a_x* (ibja_fly(X) + ijab_fly(X))
            ABX = ABX.reshape(A_size, -1)

            return ABX

        return TDA_mv, TDDFT_mv

    def gen_vind(self):
        mf = self.mf
        mo_occ = mf.mo_occ
        mol = self.mol
        un_ortho_C_matrix = mf.mo_coeff

        n_occ = len(np.where(mo_occ > 0)[0])
        n_vir = len(np.where(mo_occ == 0)[0])
        print('n_occ =', n_occ)
        print('n_vir =', n_vir)
        mo_energy = mf.mo_energy
        # print('type(mo_energy)',type(mo_energy))
        '''KS orbital energy difference, i - a
        '''
        vir = mo_energy[n_occ:].reshape(1,n_vir)
        occ = mo_energy[:n_occ].reshape(n_occ,1)
        delta_hdiag = np.repeat(vir, n_occ, axis=0) - np.repeat(occ, n_vir, axis=1)
        hdiag = delta_hdiag.reshape(n_occ*n_vir)
        Hdiag = np.vstack((hdiag, hdiag))
        Hdiag = Hdiag.reshape(-1)
        # print(hdiag)
        # print(Hdiag)

        alpha_RSH=None
        beta_RSH=None
        if mf.xc in parameter.RSH_F:
            '''
            actually I only have parameter for wb97x
            wb97x
            '''
            a_x = 1
            alpha_RSH=0.157706
            beta_RSH=0.842294
        else:
            a_x = parameter.Func_ax[mf.xc]

        auxmol_cl = self.gen_auxmol(U=self.theta, add_p=self.add_p, full_fitting=False)
        auxmol_ex = self.gen_auxmol(U=self.theta, add_p=self.add_p, full_fitting=False)

        '''
        the 2c2e and 3c2e integrals with/without RSH
        (ij|ab) -> alpha*(ij|1/r|ab)  + beta*(ij|erf(oemga)/r|ab)
        '''
        eri2c_cl, eri3c_cl, eri2c_ex, eri3c_ex = self.gen_electron_int(mol=mol,
                                                                 auxmol_cl=auxmol_cl,
                                                                 auxmol_ex=auxmol_ex,
                                                                 alpha_RSH=alpha_RSH,
                                                                 beta_RSH=beta_RSH)

        print('eri2c_cl.shape', eri2c_cl.shape)
        print('eri3c_cl.shape', eri3c_cl.shape)

        '''
        (ia|jb) tensors for coulomb always have no RSH,  have sp orbit
        (ij|ab) tensors for exchange might have RSH, might have sp orbit
        '''
        GAMMA_ia_cl, GAMMA_ia_B_cl = self.gen_GAMMA(eri2c=eri2c_cl,
                                                    eri3c=eri3c_cl,
                                                    n_occ=n_occ,
                                        un_ortho_C_matrix=un_ortho_C_matrix,
                                                    calc='coulomb')

        GAMMA_ia_ex, GAMMA_ia_B_ex, GAMMA_ab_ex, GAMMA_ij_B_ex = self.gen_GAMMA(
                                                        eri2c=eri2c_ex,
                                                        eri3c=eri3c_ex,
                                                        n_occ=n_occ,
                                            un_ortho_C_matrix=un_ortho_C_matrix,
                                                        calc='exchange')

        '''(pq|rs)V
        '''

        delta_hdiag_fly = self.gen_delta_hdiag_fly(delta_hdiag)

        iajb_fly = self.gen_coulomb(GAMMA_ia=GAMMA_ia_cl,
                                  GAMMA_ia_B=GAMMA_ia_B_cl)

        ijab_fly, ibja_fly = self.gen_exchange(GAMMA_ia=GAMMA_ia_ex,
                                             GAMMA_ia_B=GAMMA_ia_B_ex,
                                               GAMMA_ab=GAMMA_ab_ex,
                                             GAMMA_ij_B=GAMMA_ij_B_ex)

        TDA_mv, TDDFT_mv = self.gen_mv_fly(delta_hdiag_fly=delta_hdiag_fly,
                                                iajb_fly=iajb_fly,
                                                ijab_fly=ijab_fly,
                                                ibja_fly=ibja_fly,
                                                     a_x=a_x,
                                                   n_occ=n_occ,
                                                   n_vir=n_vir,
                                                   A_size=n_occ*n_vir)

        def TDA_vind(V):
            '''
            return AX
            '''
            V = np.asarray(V)
            return TDA_mv(V.T).T

        def TDDFT_vind(U):
            # print('U.shape',U.shape)
            X = U[:,:n_occ*n_vir].T
            Y = U[:,n_occ*n_vir:].T
            U1, U2 = TDDFT_mv(X, Y)
            U = np.vstack((U1, U2)).T
            return U

        return TDA_mv, TDDFT_mv, TDA_vind, TDDFT_vind, hdiag, Hdiag

    def kernel_TDA(self):
        TDA_mv, TDDFT_mv, TDA_vind, TDDFT_vind, hdiag, Hdiag = self.gen_vind()
        initial = TDA_diag_initial_guess(self.nroots, hdiag).T
        converged, e, amps = davidson1(
                      aop=TDA_vind, x0=initial, precond=hdiag,
                      tol=self.conv_tol,
                      nroots=self.nroots, lindep=1e-14,
                      max_cycle=35,
                      max_space=10000)
        e*=parameter.Hartree_to_eV
        return converged, e, amps

    # def kernel_TDDFT(self):
    #     TDA_mv, TDDFT_mv, TDA_vind, TDDFT_vind, hdiag, Hdiag = self.gen_vind()
    #
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

    def kernel_TDDFT(self):
        TDA_mv, TDDFT_mv, TDA_vind, TDDFT_vind, hdiag, Hdiag = self.gen_vind()
        name = 'TDDFT-ris'
        if self.pyscf_TDDFT_vind:
            name = 'TDDFT-abinitio'
            def TDDFT_mv(X, Y):
                '''convert pyscf style (bra) to my style (ket)
                return AX + BY and AY + BX'''
                XY = np.vstack((X,Y)).T
                U = self.pyscf_TDDFT_vind(XY)
                A_size = U.shape[1]//2
                U1 = U[:,:A_size].T
                U2 = -U[:,A_size:].T
                return U1, U2

        energies, X, Y = eigen_solver.TDDFT_eigen_solver(matrix_vector_product = TDDFT_mv,
                                                    hdiag = hdiag,
                                                    N_states = self.nroots,
                                                    conv_tol = self.conv_tol,
                                                    max_iter = self.max_iter)
        P = gen_P(self.mf, self.mol)
        eigen_solver.gen_spectra(energies, X+Y, P=P, name=name)
        return energies, X, Y
