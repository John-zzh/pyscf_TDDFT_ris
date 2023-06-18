from pyscf import gto, lib, dft
import numpy as np
from pyscf_TDDFT_ris import parameter, eigen_solver, math_helper
from mokit.lib.gaussian import load_mol_from_fch, mo_fch2py
from mokit.lib.rwwfn import read_eigenvalues_from_fch, read_nbf_and_nif_from_fch
np.set_printoptions(linewidth=250, threshold=np.inf)

einsum = lib.einsum

def get_mol_mf(fch_file, functional):

    mol = load_mol_from_fch(fch_file)
    mol.build()
    print('======= Molecular Coordinates ========')
    print(mol.atom)

    [print(k, v) for k, v in mol.basis.items()]
    mf = dft.RKS(mol)
    mf.mo_coeff = mo_fch2py(fch_file)
    nbf, nif = read_nbf_and_nif_from_fch(fch_file)
    mf.mo_energy = read_eigenvalues_from_fch(fch_file, nif=nif, ab='alpha')
    nocc = mol.nelectron // 2
    mf.mo_occ = np.asarray([2] * nocc + [0] * (nbf - nocc))
    mf.xc = functional
    return mol, mf

def gen_P(mf, mol):
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:,~occidx]
    if mol.cart:
        tag = '_cart'
    else:
        tag = '_sph'
    int_r= mol.intor_symmetric('int1e_r'+tag)
    P = lib.einsum("xpq,pi,qa->iax", int_r, orbo, orbv.conj())
    P = P.reshape(-1,3)
    return P

class TDDFT_ris(object):
    def __init__(self, mf, mol, theta=0.2,
                                add_p=False,
                             conv_tol=1e-5,
                               nroots=5,
                             max_iter=15,
                    pyscf_TDDFT_vind = None):
        '''
        add_p: whether add p orbital to aux basis
        '''
        self.mf = mf
        self.functional = mf.xc.lower()
        self.mol = mol
        self.add_p = add_p
        self.theta = theta
        self.conv_tol = conv_tol
        self.nroots = nroots
        self.max_iter = max_iter

        self.pyscf_TDDFT_vind = pyscf_TDDFT_vind

        self.RSH = False
        self.hybrid = False
        self.alpha_RSH = None
        self.beta_RSH = None

        functional = self.functional
        if functional in parameter.rsh_func.keys():
            '''
            actually I only have parameters for wb97x and cam-b3lyp
            '''
            print('use range-separated hybrid functional')
            self.RSH = True
            self.a_x = 1

            RSH_omega, alpha_RSH, beta_RSH = parameter.rsh_func[functional]
            self.RSH_omega = RSH_omega
            self.alpha_RSH = alpha_RSH
            self.beta_RSH = beta_RSH
        elif functional in parameter.hbd_func.keys():
            print('use hybrid functional')
            self.hybrid = True
            self.a_x = parameter.hbd_func[mf.xc.lower()]
        else:
            raise ValueError('functional not supported, please add parameters in parameter.py')

            
    def gen_auxmol(self, theta=0.2, add_p=False, full_fitting=False):
        print('Asigning auxiliary basis set, add p function =', add_p)
        print('The exponent alpha set as theta/R^2 ')
        print('global parameter theta =', theta)
        '''
        parse_arg = False 
        turns off PySCF built-in parsing function
        '''
        auxmol = gto.M(atom=self.mol.atom, parse_arg = False)
        auxmol_basis_keys = self.mol._basis.keys()
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
            exponent alpha = 1/R^2 * theta
            '''
            exp = parameter.ris_exp[atom] * theta
            if atom != 'H' and add_p == True:
                aux_basis[atom_index] = [[0, [exp, 1.0]],[1, [exp, 1.0]]]
            else:
                aux_basis[atom_index] = [[0, [exp, 1.0]]]

        auxmol.basis = aux_basis
        auxmol.cart = self.mol.cart
        # print('=====================')
        auxmol.build()
        # print('=====================')
        [print(k, v) for k, v in auxmol._basis.items()]

        return auxmol


    def gen_2c_3c(self, mol, auxmol, RSH_omega=0):

        '''
        Total number of contracted GTOs for the mole and auxmol object
        '''
        nao = mol.nao_nr()
        naux = auxmol.nao_nr()
        # print('nao,naux', nao,naux)
        # print('mol.cart',mol.cart)
        # print('auxmol.cart',mol.cart)
        mol.set_range_coulomb(RSH_omega)
        auxmol.set_range_coulomb(RSH_omega)

        if mol.cart:
            tag = '_cart'
        else:
            tag = '_sph'
        print('cartesian or spherical electron integral =',tag)
        '''
        (pq|rs) = Σ_PQ (pq|P)(P|Q)^-1(Q|rs)
        2 center 2 electron integral (P|Q)
        N_auxbf * N_auxbf
        '''
        eri2c = auxmol.intor('int2c2e'+tag)
        # print(eri2c)
        '''
        3 center 2 electron integral (pq|P)
        N_bf * N_bf * N_auxbf
        '''
        pmol = mol + auxmol
        pmol.cart = self.mol.cart
        # print('auxmol.nbas',auxmol.nbas)
        print('auxmol.cart =',mol.cart)
        # print('pmol.nao_nr()',pmol.nao_nr())
        print('pmol.cart',pmol.cart)

        eri3c = pmol.intor('int3c2e'+tag,
                            shls_slice=(0,mol.nbas,0,mol.nbas,
                            mol.nbas,mol.nbas+auxmol.nbas))
        
        print('eri2c.shape', eri2c.shape)
        print('eri3c.shape', eri3c.shape)
        # print(eri3c)
        return eri2c, eri3c

    def gen_electron_int(self, mol, auxmol, alpha_RSH=None, beta_RSH=None):

        eri2c_cl, eri3c_cl = self.gen_2c_3c(mol=mol, auxmol=auxmol, RSH_omega=0)
        eri2c_ex = eri2c_cl 
        eri3c_ex = eri3c_cl

        if alpha_RSH and beta_RSH:
            '''
            in the RSH functional, the Exchange electron integral splits into two parts
            (ij|ab) = (ij|1-(alpha + beta*erf(omega))/r|ab) + (ij|alpha + beta*erf(omega)/r|ab)
            -- The first part (ij|1-(alpha + beta*erf(omega))/r|ab) is short range, 
                treated by the DFT XC functional, thus not considered here
            -- The second part is long range 
                (ij|alpha + beta*erf(omega)/r|ab) = alpha (ij|r|ab) + beta*(ij|erf(omega)/r|ab)
            '''            
            print('2c2e and 3c2e for RSH RI-K (ij|ab)')
            eri2c_erf, eri3c_erf = self.gen_2c_3c(mol=mol, auxmol=auxmol, RSH_omega=self.RSH_omega)
            eri2c_ex = alpha_RSH*eri2c_ex + beta_RSH*eri2c_erf
            eri3c_ex = alpha_RSH*eri3c_ex + beta_RSH*eri3c_erf
        else:
            print('2c2e and 3c2e for RI-JK')
            
        return eri2c_cl, eri3c_cl, eri2c_ex, eri3c_ex

    def gen_B(self, eri2c, eri3c, n_occ, mo_coeff, calc=None):
        '''
        (P|Q)^-1 = LL^T
        B_pq^P = C_u^p C_v^q \sum_Q (uv|Q)L_Q
        '''
        # print('eri3c.shape', eri3c.shape)
        # L = np.linalg.cholesky(np.linalg.inv(eri2c))
        e, v = np.linalg.eigh(np.linalg.inv(eri2c))
        L = np.dot(v, np.sqrt(np.diag(e)))
        uvQL = einsum("uvQ,QP->uvP", eri3c, L)
        B = einsum("up,vq,uvP->pqP", mo_coeff, mo_coeff, uvQL)

        '''
                     n_occ          n_vir
                -|-------------||-------------|
                 |             ||             |
           n_occ |   B_ij      ||    B_ia     |
                 |             ||             |
                 |             ||             |
                =|=============||=============|
                 |             ||             |
           n_vir |             ||    B_ab     |
                 |             ||             |
                 |             ||             |
                -|-------------||-------------|
        '''


        B_ia = math_helper.copy_array(B[:n_occ,n_occ:,:])
        # print('B_ia.shape', B_ia.shape)

        if calc == 'both' or calc == 'exchange_only':
            '''
            For common bybrid DFT, exchange and coulomb term use same set of B matrix
            For range-seperated bybrid DFT, (ij|ab) and (ib|ja) use different 
            set of B matrix than (ia|jb), because of the RSH eri2c and eri3c
            '''
            B_ij = math_helper.copy_array(B[:n_occ,:n_occ,:])
            B_ab = math_helper.copy_array(B[n_occ:,n_occ:,:])
            return B_ia, B_ij, B_ab
        
        elif calc == 'coulomb_only':
            '''(ia|jb) coulomb term'''
            return B_ia

    '''
    Coulomb type integral has no RSH,
    Exchange type integral might have RSH
    '''
    def gen_coulomb(self, B_ia):
        def iajb_fly(V):
            '''
            (ia|jb) = Σ_Pjb (B_ia^P B_jb^P V_jb^m)
                    = Σ_P [ B_ia^P Σ_jb(B_jb^P V_jb^m) ]
            '''
            B_jb_V = einsum("jbP,jbm->Pm", B_ia, V)
            iajb_V = einsum("iaP,Pm->iam", B_ia, B_jb_V)
            return iajb_V
        return iajb_fly

    def gen_exchange(self, B_ia, B_ij, B_ab):
        def ijab_fly(V):
            '''
            (ij|ab) = Σ_Pjb (B_ij^P B_ab^P V_jb^m)
                    = Σ_P [B_ij^P Σ_jb(B_ab^P V_jb^m)]
            '''
            B_ab_V = einsum("abP,jbm->jPam", B_ab, V)
            ijab_V = einsum("ijP,jPam->iam", B_ij, B_ab_V)
            return ijab_V

        def ibja_fly(V):
            '''
            the exchange (ib|ja) in B matrix
            (ib|ja) = Σ_Pjb (B_ib^P B_ja^P V_jb^m)
                    = Σ_P [B_ja^P Σ_jb(B_ib^P V_jb^m)]           
            '''
            B_ib_V = einsum("ibP,jbm->Pijm", B_ia, V)
            ibja_V = einsum("jaP,Pijm->iam", B_ia, B_ib_V)
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
                                         n_occ, n_vir):
        A_size=n_occ*n_vir
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
            it can save one (ia|jb)V tensor contraction compared to directly computing AX+BY and AY+BX

            (A+B)V = delta_fly(V) + 4*iajb_fly(V) - a_x*[ibja_fly(V) + ijab_fly(V)]
            (A-B)V = delta_fly(V) + a_x[ibja_fly(V) - ijab_fly(V)]
            for RSH, a_x = 1, because the exchange component is defined by alpha+beta
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

        # def TDDFT_spolar_mv(X):

        #     ''' for RSH, a_x=1
        #         (A+B)X = delta_fly(V) + 4*iajb_fly(V) - a_x*[ijab_fly(V) + ibja_fly(V)]
        #     '''
        #     X = X.reshape(n_occ, n_vir, -1)

        #     ABX = delta_hdiag_fly(X) + 4*iajb_fly(X) - a_x* (ibja_fly(X) + ijab_fly(X))
        #     ABX = ABX.reshape(A_size, -1)

        #     return ABX

        return TDA_mv, TDDFT_mv

    def gen_vind(self):
        mf = self.mf
        mo_occ = mf.mo_occ
        mol = self.mol
        mo_coeff = mf.mo_coeff

        n_occ = len(np.where(mo_occ > 0)[0])
        n_vir = len(np.where(mo_occ == 0)[0])
        # print('n_occ =', n_occ)
        # print('n_vir =', n_vir)
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

        auxmol = self.gen_auxmol(theta=self.theta, add_p=self.add_p, full_fitting=False)

        '''
        the 2c2e and 3c2e integrals with/without RSH
        (ij|ab) = (ij|1-(alpha + beta*erf(omega))/r|ab)  + (ij|alpha + beta*erf(omega)/r|ab)
        short-range part (ij|1-(alpha + beta*erf(omega))/r|ab) is treated by the DFT XC functional, thus not considered here
        long-range part  (ij|alpha + beta*erf(omega)/r|ab) = alpha (ij|r|ab) + beta*(ij|erf(omega)/r|ab)
        '''
        # eri2c_cl, eri3c_cl, eri2c_ex, eri3c_ex = self.gen_electron_int(mol=mol,
        #                                                             auxmol=auxmol,
        #                                                          alpha_RSH=alpha_RSH,
        #                                                           beta_RSH=beta_RSH)
        if self.hybrid == True:
            
            eri2c, eri3c = self.gen_2c_3c(mol=mol, auxmol=auxmol, RSH_omega=0)
            B_ia, B_ij, B_ab = self.gen_B(eri2c=eri2c, 
                                          eri3c=eri3c, 
                                          n_occ=n_occ, 
                                          mo_coeff=mo_coeff,
                                          calc='both')
            B_ia_cl = B_ia
            B_ia_ex, B_ij_ex, B_ab_ex = B_ia, B_ij, B_ab

        elif self.RSH == True:
            
            alpha_RSH = self.alpha_RSH
            beta_RSH = self.beta_RSH
            eri2c_cl, eri3c_cl, eri2c_ex, eri3c_ex = self.gen_electron_int(mol=mol, 
                                                                           auxmol=auxmol, 
                                                                           alpha_RSH=alpha_RSH, 
                                                                           beta_RSH=beta_RSH)       
            B_ia_cl = self.gen_B(eri2c=eri2c_cl, 
                                 eri3c=eri3c_cl, 
                                 n_occ=n_occ, 
                                 mo_coeff=mo_coeff,
                                 calc='coulomb_only')
            
            B_ia_ex, B_ij_ex, B_ab_ex = self.gen_B(eri2c=eri2c_ex, 
                                                   eri3c=eri3c_ex, 
                                                   n_occ=n_occ, 
                                                   mo_coeff=mo_coeff,
                                                   calc='exchange_only')           

        '''
        (ia|jb) tensors for coulomb always have no RSH,  
        (ij|ab) tensors for exchange might have RSH
        '''

        '''
        functions of (pq|rs)V
        '''
        delta_hdiag_fly = self.gen_delta_hdiag_fly(delta_hdiag)

        iajb_fly = self.gen_coulomb(B_ia=B_ia_cl)
        ijab_fly, ibja_fly = self.gen_exchange(B_ia=B_ia_ex,
                                               B_ij=B_ij_ex,
                                               B_ab=B_ab_ex)
        a_x = self.a_x
        TDA_mv, TDDFT_mv = self.gen_mv_fly(delta_hdiag_fly=delta_hdiag_fly,
                                                iajb_fly=iajb_fly,
                                                ijab_fly=ijab_fly,
                                                ibja_fly=ibja_fly,
                                                     a_x=a_x,
                                                   n_occ=n_occ,
                                                   n_vir=n_vir)

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
        from pyscf.lib import davidson1
        _, _, TDA_vind, _, hdiag, _ = self.gen_vind()

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
        
        initial = TDA_diag_initial_guess(self.nroots, hdiag).T
        converged, e, amps = davidson1(
                      aop=TDA_vind, x0=initial, precond=hdiag,
                      tol=self.conv_tol,
                      nroots=self.nroots, lindep=1e-14,
                      max_cycle=35,
                      max_space=1000)
        e*=parameter.Hartree_to_eV
        return converged, e, amps

    # def kernel_TDDFT(self):
        # '''
        # use pyscf davidson solver to solve TDDFT
        # but pyscf davidson solver is not relibale, so we use our own davidson solver
        # '''
    #     TDA_mv, TDDFT_mv, TDA_vind, TDDFT_vind, hdiag, Hdiag = self.gen_vind()
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

    def kernel_TDDFT(self):
        if self.pyscf_TDDFT_vind:
            '''
            invoke ab-initio TDDFT from PySCF and use our davidson solver
            '''
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
        else:
            name = 'TDDFT-ris'
            _, TDDFT_mv, _, _, hdiag, _ = self.gen_vind()
            
        energies, X, Y = eigen_solver.TDDFT_eigen_solver(matrix_vector_product = TDDFT_mv,
                                                    hdiag = hdiag,
                                                    N_states = self.nroots,
                                                    conv_tol = self.conv_tol,
                                                    max_iter = self.max_iter)
        P = gen_P(self.mf, self.mol)
        eigen_solver.gen_spectra(energies, X+Y, P=P, name=name)
        return energies, X, Y
