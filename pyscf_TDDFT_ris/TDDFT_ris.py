from pyscf import gto, lib, dft
import numpy as np
from pyscf_TDDFT_ris import parameter, eigen_solver, math_helper

np.set_printoptions(linewidth=250, threshold=np.inf)

einsum = lib.einsum


class TDDFT_ris(object):
    def __init__(self, mf: dft, 
                theta: float = 0.2,
                add_p: bool = False,
                a_x: float = None,
                omega: float = None,
                alpha: float = None,
                beta: float = None,
                conv_tol: float = 1e-5,
                nroots: int = 5,
                max_iter: int = 25,
                pyscf_TDDFT_vind: callable = None):
        '''
        add_p: whether add p orbital to aux basis
        '''
        self.mf = mf
        self.theta = theta
        self.add_p = add_p
        self.a_x = a_x
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.conv_tol = conv_tol
        self.nroots = nroots
        self.max_iter = max_iter
        self.mol = mf.mol
        self.pyscf_TDDFT_vind = pyscf_TDDFT_vind

        if hasattr(mf, 'xc'):
            functional = mf.xc.lower()
            self.functional = mf.xc
            print('Loading defult functional paramters from parameter.py.')
            if functional in parameter.rsh_func.keys():
                '''
                RSH functional, need omega, alpha, beta
                '''
                print('use range-separated hybrid functional')
                omega, alpha, beta = parameter.rsh_func[functional]
                self.a_x = 1
                self.omega = omega
                self.alpha = alpha
                self.beta = beta

            elif functional in parameter.hbd_func.keys():
                print('use hybrid functional')
                self.a_x = parameter.hbd_func[functional]

            else:
                raise ValueError(f"I do not have paramters for functional {mf.xc} yet, please either manually input HF component a_x or add parameters in the parameter.py file.")
            
        else:
            if self.a_x == None and self.omega == None and self.alpha == None and self.beta == None:
                raise ValueError('Please specify the functional name or the functional parameters')
            else:
                if a_x:
                    self.a_x = a_x
                    print("hybrid functional")
                    print(f"manually input HF component ax = {a_x}")

                elif omega and alpha and beta:
                    self.a_x = 1
                    self.omega = omega
                    self.alpha = alpha
                    self.beta = beta
                    print("range-separated hybrid functional")
                    print(f"manually input ω = {self.omega}, screening factor")
                    print(f"manually input α = {self.alpha}, fixed HF exchange contribution")
                    print(f"manually input β = {self.beta}, variable part")

                else:
                    raise ValueError('missing parameters for range-separated functional, please input (w, al, be)')





        if self.mol.cart:
            self.eri_tag = '_cart'
        else:
            self.eri_tag = '_sph'
        print('cartesian or spherical electron integral =',self.eri_tag.split('_')[1])

        if mf.mo_coeff.ndim == 2:
            self.RKS = True
            self.UKS = False
            self.n_bf = len(mf.mo_occ)
            self.n_occ = sum(mf.mo_occ>0)
            self.n_vir = self.n_bf - self.n_occ
            
        elif mf.mo_coeff.ndim == 3:
            self.RKS = False
            self.UKS = True
            self.n_bf = len(mf.mo_occ[0])
            self.n_occ_a = sum(mf.mo_occ[0]>0)
            self.n_vir_a = self.n_bf - self.n_occ_a
            self.n_occ_b = sum(mf.mo_occ[1]>0)
            self.n_vir_b = self.n_bf - self.n_occ_b
            print('n_occ for alpha spin =',self.n_occ_a)
            print('n_vir for alpha spin =',self.n_vir_a)
            print('n_occ for beta spin =',self.n_occ_b)
            print('n_vir for beta spin =',self.n_vir_b)

            
    def gen_auxmol(self, theta=0.2, add_p=False):
        print('Asigning minimal auxiliary basis set')
        if add_p:
            print('add p orbitals to auxbasis')
        print('The exponent alpha set as theta/R^2 ')
        print('global parameter theta =', theta)
        '''
        parse_arg = False 
        turns off PySCF built-in parsing function
        '''
        auxmol = gto.M(atom=self.mol.atom, 
                        parse_arg = False, 
                        spin=self.mol.spin, 
                        charge=self.mol.charge,
                        cart = self.mol.cart)
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
        auxmol.build()
        self.auxmol = auxmol
        # print('=====================')
        # [print(k, v) for k, v in auxmol._basis.items()]

        return auxmol


    def gen_eri2c_eri3c(self, mol, auxmol, omega=0):

        '''
        Total number of contracted GTOs for the mole and auxmol object
        '''
        # nao = mol.nao_nr()
        # naux = auxmol.nao_nr()

        mol.set_range_coulomb(omega)
        auxmol.set_range_coulomb(omega)

        '''
        (pq|rs) = Σ_PQ (pq|P)(P|Q)^-1(Q|rs)
        2 center 2 electron integral (P|Q)
        N_auxbf * N_auxbf
        '''
        tag = self.eri_tag 
        eri2c = auxmol.intor('int2c2e'+tag)
        # print(eri2c)
        '''
        3 center 2 electron integral (pq|P)
        N_bf * N_bf * N_auxbf
        '''
        pmol = mol + auxmol
        pmol.cart = self.mol.cart
        print('auxmol.cart =',mol.cart)

        eri3c = pmol.intor('int3c2e'+tag,
                            shls_slice=(0,mol.nbas,0,mol.nbas,
                            mol.nbas,mol.nbas+auxmol.nbas))
        
        print('Three center ERI shape', eri3c.shape)

        return eri2c, eri3c

    def gen_eri2c_eri3c_RSH(self, mol, auxmol, eri2c_ex, eri3c_ex, alpha, beta, omega):

        '''
        in the RSH functional, the Exchange ERI splits into two parts
        (ij|ab) = (ij|1-(alpha + beta*erf(omega))/r|ab) + (ij|alpha + beta*erf(omega)/r|ab)
        -- The first part (ij|1-(alpha + beta*erf(omega))/r|ab) is short range, 
            treated by the DFT XC functional, thus not considered here
        -- The second part is long range 
            (ij|alpha + beta*erf(omega)/r|ab) = alpha (ij|r|ab) + beta*(ij|erf(omega)/r|ab)
        '''            
        print('2c2e and 3c2e for RSH RI-K (ij|ab)')
        eri2c_erf, eri3c_erf = self.gen_eri2c_eri3c(mol=mol, auxmol=auxmol, omega=omega)
        eri2c_ex = alpha*eri2c_ex + beta*eri2c_erf
        eri3c_ex = alpha*eri3c_ex + beta*eri3c_erf
            
        return eri2c_ex, eri3c_ex

    def gen_uvQL(self, eri3c, eri2c):
        '''
        (P|Q)^-1 = LL^T
        uvQL = Σ_Q (uv|Q)L_Q
        '''
        # print('eri3c.shape', eri3c.shape)
        Lower = np.linalg.cholesky(np.linalg.inv(eri2c))
        # e, v = np.linalg.eigh(np.linalg.inv(eri2c))
        # L = np.dot(v, np.sqrt(np.diag(e)))
        uvQL = einsum("uvQ,QP->uvP", eri3c, Lower)
        return uvQL

    def gen_B(self, uvQL, n_occ, mo_coeff, calc=None):
        ''' B_pq^P = C_u^p C_v^q Σ_Q (uv|Q)L_Q '''
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
            B_ia_ex is for (ib|ja)
            '''
            B_ij = math_helper.copy_array(B[:n_occ,:n_occ,:])
            B_ab = math_helper.copy_array(B[n_occ:,n_occ:,:])
            return B_ia, B_ij, B_ab
        
        elif calc == 'coulomb_only':
            '''(ia|jb) coulomb term'''
            return B_ia

    def gen_B_cl_B_ex(self, mol, auxmol, uvQL, n_occ, mo_coeff, eri3c=None, eri2c=None):

        '''
        the 2c2e and 3c2e integrals with/without RSH
        (ij|ab) = (ij|1-(alpha + beta*erf(omega))/r|ab)  + (ij|alpha + beta*erf(omega)/r|ab)
        short-range part (ij|1-(alpha + beta*erf(omega))/r|ab) is treated by the DFT XC functional, thus not considered here
        long-range part  (ij|alpha + beta*erf(omega)/r|ab) = alpha (ij|r|ab) + beta*(ij|erf(omega)/r|ab)
        ''' 

        if self.a_x != 0 and self.omega == None:
            '''
            for usual hybrid functional, the Coulomb and Exchange ERI
            share the same eri2c and eri3c,
            '''
            
            B_ia, B_ij, B_ab = self.gen_B(uvQL=uvQL, 
                                          n_occ=n_occ, 
                                          mo_coeff=mo_coeff,
                                          calc='both')
            B_ia_cl = B_ia
            B_ia_ex, B_ij_ex, B_ab_ex = B_ia, B_ij, B_ab

        elif self.omega:
            '''
            for range-saparated hybrid functional, the Coulomb and Exchange ERI
            use modified eri2c and eri3c,
            eri2c -> eri2c_ex
            eri3c -> eri3c_ex
            '''            
            # eri2c_ex, eri3c_ex = self.gen_eri2c_eri3c_RSH(mol=mol,
            #                                             auxmol=auxmol,
            #                                             eri2c_ex=eri2c, 
            #                                             eri3c_ex=eri3c,
            #                                             alpha=self.alpha, 
            #                                             beta=self.beta,
            #                                             omega=self.omega) 

            print('2c2e and 3c2e for RSH RI-K (ij|ab)')
            eri2c_erf, eri3c_erf = self.gen_eri2c_eri3c(mol=mol, auxmol=auxmol, omega=self.omega)
            eri2c_ex = self.alpha*eri2c + self.beta*eri2c_erf
            eri3c_ex = self.alpha*eri3c + self.beta*eri3c_erf 

            B_ia_cl = self.gen_B(uvQL=uvQL,
                                 n_occ=n_occ, 
                                 mo_coeff=mo_coeff,
                                 calc='coulomb_only')
            uvQL_ex = self.gen_uvQL(eri2c=eri2c_ex, eri3c=eri3c_ex) 
            B_ia_ex, B_ij_ex, B_ab_ex = self.gen_B(uvQL=uvQL_ex, 
                                                   n_occ=n_occ, 
                                                   mo_coeff=mo_coeff,
                                                   calc='exchange_only')           
        return B_ia_cl, B_ia_ex, B_ij_ex, B_ab_ex

    
    def gen_hdiag_fly(self, mo_energy, n_occ, n_vir, sqrt=False):

        '''KS orbital energy difference, ε_a - ε_i
        '''
        vir = mo_energy[n_occ:].reshape(1,n_vir)
        occ = mo_energy[:n_occ].reshape(n_occ,1)
        delta_hdiag = np.repeat(vir, n_occ, axis=0) - np.repeat(occ, n_vir, axis=1)

        hdiag = delta_hdiag.reshape(n_occ*n_vir)
        # Hdiag = np.vstack((hdiag, hdiag))
        # Hdiag = Hdiag.reshape(-1)
        if sqrt == False:
            '''standard diag(A)V
               preconditioner = diag(A)
            '''
            def hdiag_fly(V):
                delta_hdiag_v = einsum("ia,iam->iam", delta_hdiag, V)
                return delta_hdiag_v
            return hdiag_fly, hdiag
        
        elif sqrt == True:
            '''diag(A)**0.5 V
               preconditioner = diag(A)**2
            '''
            delta_hdiag_sqrt = np.sqrt(delta_hdiag)
            hdiag_sq = hdiag**2
            def hdiag_sqrt_fly(V):
                delta_hdiag_v = einsum("ia,iam->iam", delta_hdiag_sqrt, V)
                return delta_hdiag_v
            return hdiag_sqrt_fly, hdiag_sq


    def gen_iajb_fly(self, B_left, B_right):
        def iajb_fly(V):
            '''
            (ia|jb) = Σ_Pjb (B_left_ia^P B_right_jb^P V_jb^m)
                    = Σ_P [ B_left_ia^P Σ_jb(B_right_jb^P V_jb^m) ]
            if B_left == B_right, then it is either 
                (1) (ia|jb) in RKS 
                or 
                (2)(ia_α|jb_α) or (ia_β|jb_β) in UKS, 
            else, 
                it is (ia_α|jb_β) or (ia_β|jb_α) in UKS
            '''
            B_right_jb_V = einsum("jbP,jbm->Pm", B_right, V)
            iajb_V = einsum("iaP,Pm->iam", B_left, B_right_jb_V)
            return iajb_V
        return iajb_fly
    
    def gen_ijab_fly(self, B_ij, B_ab):
        def ijab_fly(V):
            '''
            (ij|ab) = Σ_Pjb (B_ij^P B_ab^P V_jb^m)
                    = Σ_P [B_ij^P Σ_jb(B_ab^P V_jb^m)]
            '''
            B_ab_V = einsum("abP,jbm->jPam", B_ab, V)
            ijab_V = einsum("ijP,jPam->iam", B_ij, B_ab_V)
            return ijab_V
        return ijab_fly
    
    def gen_ibja_fly(self, B_ia):    
        def ibja_fly(V):
            '''
            the exchange (ib|ja) in B matrix
            (ib|ja) = Σ_Pjb (B_ib^P B_ja^P V_jb^m)
                    = Σ_P [B_ja^P Σ_jb(B_ib^P V_jb^m)]           
            '''
            B_ib_V = einsum("ibP,jbm->Pijm", B_ia, V)
            ibja_V = einsum("jaP,Pijm->iam", B_ia, B_ib_V)
            return ibja_V
        return ibja_fly

    #  ===========  RKS ===========
    def gen_RKS_TDA_mv(self):
        a_x = self.a_x
        n_occ = self.n_occ
        n_vir = self.n_vir
        A_size=n_occ*n_vir

        mo_coeff = self.mf.mo_coeff
        mo_energy = self.mf.mo_energy

        mol = self.mol
        auxmol = self.gen_auxmol(theta=self.theta, add_p=self.add_p)
        eri2c, eri3c = self.gen_eri2c_eri3c(mol=self.mol, auxmol=auxmol, omega=0)
        uvQL = self.gen_uvQL(eri2c=eri2c, eri3c=eri3c)

        hdiag_fly, hdiag = self.gen_hdiag_fly(mo_energy=mo_energy,  
                                              n_occ=n_occ, 
                                              n_vir=n_vir)

        if a_x != 0:
            '''hybrid RKS TDA'''
            B_ia_cl, _, B_ij_ex, B_ab_ex = self.gen_B_cl_B_ex(mol=mol,
                                                            auxmol=auxmol,
                                                            uvQL=uvQL, 
                                                            eri3c=eri3c, 
                                                            eri2c=eri2c,
                                                            n_occ=n_occ, 
                                                            mo_coeff=mo_coeff)

            iajb_fly = self.gen_iajb_fly(B_left=B_ia_cl, B_right=B_ia_cl)
            ijab_fly = self.gen_ijab_fly(B_ij=B_ij_ex, B_ab=B_ab_ex)
            
            def RKS_TDA_hybrid_mv(X):
                ''' hybrid or range-sparated hybrid, a_x > 0
                    return AX
                    AV = hdiag_fly(V) + 2*iajb_fly(V) - a_x*ijab_fly(V)
                    for RSH, a_x = 1
                '''
                # print('a_x=', a_x)
                X = X.reshape(n_occ, n_vir, -1)
                AX = hdiag_fly(X) + 2*iajb_fly(X) - a_x*ijab_fly(X)
                AX = AX.reshape(A_size, -1)
                return AX
            return RKS_TDA_hybrid_mv, hdiag
            
        elif a_x == 0:
            '''pure RKS TDA'''
            B_ia = self.gen_B(uvQL=uvQL,
                                n_occ=n_occ, 
                                mo_coeff=mo_coeff,
                                calc='coulomb_only')
            iajb_fly = self.gen_iajb_fly(B_left=B_ia, B_right=B_ia)
            def RKS_TDA_pure_mv(X):
                ''' pure functional, a_x = 0
                    return AX
                    AV = hdiag_fly(V) + 2*iajb_fly(V) 
                    for RSH, a_x = 1
                '''
                # print('a_x=', a_x)
                X = X.reshape(n_occ, n_vir, -1)
                AX = hdiag_fly(X) + 2*iajb_fly(X)
                AX = AX.reshape(A_size, -1)
                return AX

        return RKS_TDA_pure_mv, hdiag
    
    def gen_RKS_TDDFT_mv(self):  
        
        a_x = self.a_x
        n_occ = self.n_occ
        n_vir = self.n_vir
        A_size=n_occ*n_vir 

        mo_coeff = self.mf.mo_coeff
        mo_energy = self.mf.mo_energy

        mol = self.mol
        auxmol = self.gen_auxmol(theta=self.theta, add_p=self.add_p)
        eri2c, eri3c = self.gen_eri2c_eri3c(mol=self.mol, auxmol=auxmol, omega=0)
        uvQL = self.gen_uvQL(eri2c=eri2c, eri3c=eri3c)

        '''hdiag_fly will be used in both RKS and UKS'''
        hdiag_fly, hdiag = self.gen_hdiag_fly(mo_energy=mo_energy, 
                                                n_occ=n_occ, 
                                                n_vir=n_vir)
        if a_x != 0:
            '''hybrid or range-separated RKS TDDFT'''

            B_ia_cl, B_ia_ex, B_ij_ex, B_ab_ex = self.gen_B_cl_B_ex(mol=mol,
                                                                    auxmol=auxmol,
                                                                    uvQL=uvQL, 
                                                                    eri3c=eri3c, 
                                                                    eri2c=eri2c,
                                                                    n_occ=n_occ, 
                                                                    mo_coeff=mo_coeff)
            iajb_fly = self.gen_iajb_fly(B_left=B_ia_cl, B_right=B_ia_cl)
            ijab_fly = self.gen_ijab_fly(B_ij=B_ij_ex, B_ab=B_ab_ex)
            ibja_fly = self.gen_ibja_fly(B_ia=B_ia_ex)

            def RKS_TDDFT_hybrid_mv(X, Y):
                '''
                RKS
                [A B][X] = [AX+BY] = [U1]
                [B A][Y]   [AY+BX]   [U2]
                we want AX+BY and AY+BX
                instead of directly computing AX+BY and AY+BX 
                we compute (A+B)(X+Y) and (A-B)(X-Y)
                it can save one (ia|jb)V tensor contraction compared to directly computing AX+BY and AY+BX

                (A+B)V = hdiag_fly(V) + 4*iajb_fly(V) - a_x * [ ijab_fly(V) + ibja_fly(V) ]
                (A-B)V = hdiag_fly(V) - a_x * [ ijab_fly(V) - ibja_fly(V) ]
                for RSH, a_x = 1, because the exchange component is defined by alpha+beta
                '''
                X = X.reshape(n_occ, n_vir, -1)
                Y = Y.reshape(n_occ, n_vir, -1)

                XpY = X + Y
                XmY = X - Y

                ApB_XpY = hdiag_fly(XpY) + 4*iajb_fly(XpY) - a_x*(ijab_fly(XpY) + ibja_fly(XpY))

                AmB_XmY = hdiag_fly(XmY) - a_x*(ijab_fly(XmY) - ibja_fly(XmY) )

                ''' (A+B)(X+Y) = AX + BY + AY + BX   (1)
                    (A-B)(X-Y) = AX + BY - AY - BX   (2)
                    (1) + (1) /2 = AX + BY = U1
                    (1) - (2) /2 = AY + BX = U2
                '''
                U1 = (ApB_XpY + AmB_XmY)/2
                U2 = (ApB_XpY - AmB_XmY)/2

                U1 = U1.reshape(A_size,-1)
                U2 = U2.reshape(A_size,-1)

                return U1, U2
            return RKS_TDDFT_hybrid_mv, hdiag
        
        elif a_x == 0:
            '''pure RKS TDDFT'''
            hdiag_sqrt_fly, hdiag_sq = self.gen_hdiag_fly(mo_energy=mo_energy, 
                                                          n_occ=n_occ, 
                                                          n_vir=n_vir,
                                                          sqrt=True)
            B_ia = self.gen_B(uvQL=uvQL,
                            n_occ=n_occ, 
                            mo_coeff=mo_coeff,
                            calc='coulomb_only')
            iajb_fly = self.gen_iajb_fly(B_left=B_ia, B_right=B_ia)

            def RKS_TDDFT_pure_mv(Z):
                '''(A-B)^1/2(A+B)(A-B)^1/2 Z = Z w^2
                                        MZ = Z w^2
                    X+Y = (A-B)^1/2 Z
                    A+B = hdiag_fly(V) + 4*iajb_fly(V)
                    (A-B)^1/2 = hdiag_sqrt_fly(V)
                '''
                Z = Z.reshape(n_occ, n_vir, -1)
                AmB_sqrt_V = hdiag_sqrt_fly(Z)
                ApB_AmB_sqrt_V = hdiag_fly(AmB_sqrt_V) + 4*iajb_fly(AmB_sqrt_V)
                MZ = hdiag_sqrt_fly(ApB_AmB_sqrt_V)
                MZ = MZ.reshape(A_size, -1)
                return MZ
            
            return RKS_TDDFT_pure_mv, hdiag_sq
        
    #  ===========  UKS ===========
    def gen_UKS_TDA_mv(self):
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
        auxmol = self.gen_auxmol(theta=self.theta, add_p=self.add_p)
        eri2c, eri3c = self.gen_eri2c_eri3c(mol=self.mol, auxmol=auxmol, omega=0)
        uvQL = self.gen_uvQL(eri2c=eri2c, eri3c=eri3c)

        hdiag_a_fly, hdiag_a = self.gen_hdiag_fly(mo_energy=mo_energy[0], n_occ=n_occ_a, n_vir=n_vir_a)
        hdiag_b_fly, hdiag_b = self.gen_hdiag_fly(mo_energy=mo_energy[1], n_occ=n_occ_b, n_vir=n_vir_b)
        hdiag = np.vstack((hdiag_a.reshape(-1,1), hdiag_b.reshape(-1,1))).reshape(-1)

        if a_x != 0:
            ''' UKS TDA hybrid '''
            B_ia_cl_alpha, _, B_ij_ex_alpha, B_ab_ex_alpha = self.gen_B_cl_B_ex(mol=mol,
                                                                                auxmol=auxmol,
                                                                                uvQL=uvQL, 
                                                                                eri3c=eri3c, 
                                                                                eri2c=eri2c,
                                                                                n_occ=n_occ_a, 
                                                                                mo_coeff=mo_coeff[0])
            
            B_ia_cl_beta, _, B_ij_ex_beta, B_ab_ex_beta  = self.gen_B_cl_B_ex(mol=mol,
                                                                              auxmol=auxmol,
                                                                              uvQL=uvQL,
                                                                              eri3c=eri3c, 
                                                                              eri2c=eri2c,
                                                                              n_occ=n_occ_b,
                                                                              mo_coeff=mo_coeff[1])

            iajb_aa_fly = self.gen_iajb_fly(B_left=B_ia_cl_alpha, B_right=B_ia_cl_alpha)
            iajb_ab_fly = self.gen_iajb_fly(B_left=B_ia_cl_alpha, B_right=B_ia_cl_beta)
            iajb_ba_fly = self.gen_iajb_fly(B_left=B_ia_cl_beta,  B_right=B_ia_cl_alpha)
            iajb_bb_fly = self.gen_iajb_fly(B_left=B_ia_cl_beta,  B_right=B_ia_cl_beta)

            ijab_aa_fly = self.gen_ijab_fly(B_ij=B_ij_ex_alpha, B_ab=B_ab_ex_alpha)
            ijab_bb_fly = self.gen_ijab_fly(B_ij=B_ij_ex_beta,  B_ab=B_ab_ex_beta)
            
            def UKS_TDA_hybrid_mv(X):
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

                Aαα Xα = hdiag_fly(Xα) + iajb_aa_fly(Xα) - a_x * ijab_aa_fly(Xα) 
                Aββ Xβ = hdiag_fly(Xβ) + iajb_bb_fly(Xβ) - a_x * ijab_bb_fly(Xβ)
                Aαβ Xβ = iajb_ab_fly(Xβ)
                Aβα Xα = iajb_ba_fly(Xα)
                '''
                X_a = X[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                X_b = X[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)

                Aaa_Xa = hdiag_a_fly(X_a) + iajb_aa_fly(X_a) - a_x * ijab_aa_fly(X_a) 
                Aab_Xb = iajb_ab_fly(X_b)

                Aba_Xa = iajb_ba_fly(X_a)
                Abb_Xb = hdiag_b_fly(X_b) + iajb_bb_fly(X_b) - a_x * ijab_bb_fly(X_b)
                
                U_a = (Aaa_Xa + Aab_Xb).reshape(A_aa_size,-1)
                U_b = (Aba_Xa + Abb_Xb).reshape(A_bb_size,-1)

                U = np.vstack((U_a, U_b))
                return U
            return UKS_TDA_hybrid_mv, hdiag

        elif a_x == 0:
            ''' UKS TDA pure '''
            B_ia_alpha = self.gen_B(uvQL=uvQL,
                                    n_occ=n_occ_a, 
                                    mo_coeff=mo_coeff[0],
                                    calc='coulomb_only')
            B_ia_beta = self.gen_B(uvQL=uvQL,
                                    n_occ=n_occ_b, 
                                    mo_coeff=mo_coeff[1],
                                    calc='coulomb_only')
            
            iajb_aa_fly = self.gen_iajb_fly(B_left=B_ia_alpha, B_right=B_ia_alpha)
            iajb_ab_fly = self.gen_iajb_fly(B_left=B_ia_alpha, B_right=B_ia_beta)
            iajb_ba_fly = self.gen_iajb_fly(B_left=B_ia_beta,  B_right=B_ia_alpha)
            iajb_bb_fly = self.gen_iajb_fly(B_left=B_ia_beta,  B_right=B_ia_beta)

            def UKS_TDA_pure_mv(X):
                '''
                Aαα Xα = hdiag_fly(Xα) + iajb_aa_fly(Xα)  
                Aββ Xβ = hdiag_fly(Xβ) + iajb_bb_fly(Xβ) 
                Aαβ Xβ = iajb_ab_fly(Xβ)
                Aβα Xα = iajb_ba_fly(Xα)
                '''
                X_a = X[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                X_b = X[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)

                Aaa_Xa = hdiag_a_fly(X_a) + iajb_aa_fly(X_a)
                Aab_Xb = iajb_ab_fly(X_b)

                Aba_Xa = iajb_ba_fly(X_a)
                Abb_Xb = hdiag_b_fly(X_b) + iajb_bb_fly(X_b) 
                
                U_a = (Aaa_Xa + Aab_Xb).reshape(A_aa_size,-1)
                U_b = (Aba_Xa + Abb_Xb).reshape(A_bb_size,-1)

                U = np.vstack((U_a, U_b))
                return U
            return UKS_TDA_pure_mv, hdiag

    def gen_UKS_TDDFT_mv(self):
        
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
        auxmol = self.gen_auxmol(theta=self.theta, add_p=self.add_p)
        eri2c, eri3c = self.gen_eri2c_eri3c(mol=self.mol, auxmol=auxmol, omega=0)
        uvQL = self.gen_uvQL(eri2c=eri2c, eri3c=eri3c)
        '''
        _aa_fly means alpha-alpha spin
        _ab_fly means alpha-beta spin
        B_ia_alpha means B_ia matrix for alpha spin
        B_ia_beta means B_ia matrix for beta spin
        '''

        hdiag_a_fly, hdiag_a = self.gen_hdiag_fly(mo_energy=mo_energy[0], n_occ=n_occ_a, n_vir=n_vir_a)
        hdiag_b_fly, hdiag_b = self.gen_hdiag_fly(mo_energy=mo_energy[1], n_occ=n_occ_b, n_vir=n_vir_b)
        hdiag = np.vstack((hdiag_a.reshape(-1,1), hdiag_b.reshape(-1,1))).reshape(-1)

        if a_x != 0:
            B_ia_cl_alpha, B_ia_ex_alpha, B_ij_ex_alpha, B_ab_ex_alpha = self.gen_B_cl_B_ex(mol=mol,
                                                                                            auxmol=auxmol,
                                                                                            uvQL=uvQL, 
                                                                                            eri3c=eri3c, 
                                                                                            eri2c=eri2c,
                                                                                            n_occ=n_occ_a, 
                                                                                            mo_coeff=mo_coeff[0])
            
            B_ia_cl_beta,  B_ia_ex_beta,  B_ij_ex_beta,  B_ab_ex_beta  = self.gen_B_cl_B_ex(mol=mol,
                                                                                            auxmol=auxmol,
                                                                                            uvQL=uvQL,
                                                                                            eri3c=eri3c, 
                                                                                            eri2c=eri2c,
                                                                                            n_occ=n_occ_b,
                                                                                            mo_coeff=mo_coeff[1])

            iajb_aa_fly = self.gen_iajb_fly(B_left=B_ia_cl_alpha, B_right=B_ia_cl_alpha)
            iajb_ab_fly = self.gen_iajb_fly(B_left=B_ia_cl_alpha, B_right=B_ia_cl_beta)
            iajb_ba_fly = self.gen_iajb_fly(B_left=B_ia_cl_beta,  B_right=B_ia_cl_alpha)
            iajb_bb_fly = self.gen_iajb_fly(B_left=B_ia_cl_beta,  B_right=B_ia_cl_beta)

            ijab_aa_fly = self.gen_ijab_fly(B_ij=B_ij_ex_alpha, B_ab=B_ab_ex_alpha)
            ijab_bb_fly = self.gen_ijab_fly(B_ij=B_ij_ex_beta,  B_ab=B_ab_ex_beta)
            
            ibja_aa_fly = self.gen_ibja_fly(B_ia=B_ia_ex_alpha)
            ibja_bb_fly = self.gen_ibja_fly(B_ia=B_ia_ex_beta)

            def UKS_TDDFT_hybrid_mv(X,Y):
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
                (A+B)αα Vα = hdiag_fly(Vα) + 2*iaαjbα_fly(Vα) - a_x*[ijαabα_fly(Vα) + ibαjaα_fly(Vα)]
                (A+B)αβ Vβ = 2*iaαjbβ_fly(Vβ) 

                V:= X-Y
                (A-B)αα Vα = hdiag_fly(Vα) - a_x*[ijαabα_fly(Vα) - ibαjaα_fly(Vα)]
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
                ApB_XpY_aa = hdiag_a_fly(XpY_a) + 2*iajb_aa_fly(XpY_a) - a_x*(ijab_aa_fly(XpY_a) + ibja_aa_fly(XpY_a))
                '''(A+B)bb(X+Y)b'''
                ApB_XpY_bb = hdiag_b_fly(XpY_b) + 2*iajb_bb_fly(XpY_b) - a_x*(ijab_bb_fly(XpY_b) + ibja_bb_fly(XpY_b))           
                '''(A+B)ab(X+Y)b'''
                ApB_XpY_ab = 2*iajb_ab_fly(XpY_b)
                '''(A+B)ba(X+Y)a'''
                ApB_XpY_ba = 2*iajb_ba_fly(XpY_a)

                '''============== (A-B) (X-Y) ================'''
                '''(A-B)aa(X-Y)a'''
                AmB_XmY_aa = hdiag_a_fly(XmY_a) - a_x*(ijab_aa_fly(XmY_a) - ibja_aa_fly(XmY_a))
                '''(A-B)bb(X-Y)b'''
                AmB_XmY_bb = hdiag_b_fly(XmY_b) - a_x*(ijab_bb_fly(XmY_b) - ibja_bb_fly(XmY_b))  

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

            return UKS_TDDFT_hybrid_mv, hdiag

        elif a_x == 0:
            ''' UKS TDDFT pure '''

            hdiag_a_sqrt_fly, hdiag_a_sq = self.gen_hdiag_fly(mo_energy=mo_energy[0], 
                                                          n_occ=n_occ_a, 
                                                          n_vir=n_vir_a,
                                                          sqrt=True)
            hdiag_b_sqrt_fly, hdiag_b_sq = self.gen_hdiag_fly(mo_energy=mo_energy[1], 
                                                          n_occ=n_occ_b, 
                                                          n_vir=n_vir_b,
                                                          sqrt=True)
            '''hdiag_sq: preconditioner'''
            hdiag_sq = np.vstack((hdiag_a_sq.reshape(-1,1), hdiag_b_sq.reshape(-1,1))).reshape(-1)

            B_ia_alpha = self.gen_B(uvQL=uvQL,
                                    n_occ=n_occ_a, 
                                    mo_coeff=mo_coeff[0],
                                    calc='coulomb_only')
            B_ia_beta = self.gen_B(uvQL=uvQL,
                                    n_occ=n_occ_b, 
                                    mo_coeff=mo_coeff[1],
                                    calc='coulomb_only')
                       
            iajb_aa_fly = self.gen_iajb_fly(B_left=B_ia_alpha, B_right=B_ia_alpha)
            iajb_ab_fly = self.gen_iajb_fly(B_left=B_ia_alpha, B_right=B_ia_beta)
            iajb_ba_fly = self.gen_iajb_fly(B_left=B_ia_beta,  B_right=B_ia_alpha)
            iajb_bb_fly = self.gen_iajb_fly(B_left=B_ia_beta,  B_right=B_ia_beta)

            def UKS_TDDFT_pure_mv(Z):
                '''       MZ = Z w^2
                    M = (A-B)^1/2(A+B)(A-B)^1/2
                    Z = (A-B)^1/2(X-Y)

                    X+Y = (A-B)^1/2 Z * 1/w
                    A+B = hdiag_fly(V) + 4*iajb_fly(V)
                    (A-B)^1/2 = hdiag_sqrt_fly(V)


                    M =  [ (A-B)^1/2αα    0   ] [ (A+B)αα (A+B)αβ ] [ (A-B)^1/2αα    0   ]            Z = [ Zα ]  
                         [    0   (A-B)^1/2ββ ] [ (A+B)βα (A+B)ββ ] [    0   (A-B)^1/2ββ ]                [ Zβ ]
                '''
                Z_a = Z[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                Z_b = Z[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)

                AmB_aa_sqrt_Z_a = hdiag_a_sqrt_fly(Z_a)
                AmB_bb_sqrt_Z_b = hdiag_b_sqrt_fly(Z_b)

                ApB_aa_sqrt_V = hdiag_a_fly(AmB_aa_sqrt_Z_a) + 2*iajb_aa_fly(AmB_aa_sqrt_Z_a)
                ApB_ab_sqrt_V = 2*iajb_ab_fly(AmB_bb_sqrt_Z_b)
                ApB_ba_sqrt_V = 2*iajb_ba_fly(AmB_aa_sqrt_Z_a)
                ApB_bb_sqrt_V = hdiag_b_fly(AmB_bb_sqrt_Z_b) + 2*iajb_bb_fly(AmB_bb_sqrt_Z_b)

                MZ_a = hdiag_a_sqrt_fly(ApB_aa_sqrt_V + ApB_ab_sqrt_V).reshape(A_aa_size, -1)
                MZ_b = hdiag_b_sqrt_fly(ApB_ba_sqrt_V + ApB_bb_sqrt_V).reshape(A_bb_size, -1)

                MZ = np.vstack((MZ_a, MZ_b))
                # print(MZ.shape)
                return MZ
        
            return UKS_TDDFT_pure_mv, hdiag_sq

        # def TDDFT_spolar_mv(X):

        #     ''' for RSH, a_x=1
        #         (A+B)X = hdiag_fly(V) + 4*iajb_fly(V) - a_x*[ijab_fly(V) + ibja_fly(V)]
        #     '''
        #     X = X.reshape(n_occ, n_vir, -1)

        #     ABX = hdiag_fly(X) + 4*iajb_fly(X) - a_x* (ibja_fly(X) + ijab_fly(X))
        #     ABX = ABX.reshape(A_size, -1)

        #     return ABX
        
    # def gen_vind(self, TDA_mv, TDDFT_mv):
    #     '''
    #     _vind is pyscf style, to feed pyscf eigensovler
    #     _mv is my style, to feed my eigensovler
    #     '''
    #     def TDA_vind(V):
    #         '''
    #         return AX
    #         '''
    #         V = np.asarray(V)
    #         return TDA_mv(V.T).T

    #     def TDDFT_vind(U):
    #         # print('U.shape',U.shape)
    #         X = U[:,:n_occ*n_vir].T
    #         Y = U[:,n_occ*n_vir:].T
    #         U1, U2 = TDDFT_mv(X, Y)
    #         U = np.vstack((U1, U2)).T
    #         return U

    #     return TDA_vind, TDDFT_vind

    def gen_P(self, int_r, mo_coeff, mo_occ):
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

    def gen_RKS_P(self):
        '''
        transition dipole P
        '''
        tag = self.eri_tag
        int_r = self.mol.intor_symmetric('int1e_r'+tag)
        mo_occ = self.mf.mo_occ
        mo_coeff = self.mf.mo_coeff

        P = self.gen_P(int_r=int_r, mo_coeff=mo_coeff, mo_occ=mo_occ)
        P = P.reshape(-1,3)
        return P

    def gen_UKS_P(self):
        '''
        transition dipole,
        P = [P_α]
            [P_β]
        '''
        tag = self.eri_tag
        int_r = self.mol.intor_symmetric('int1e_r'+tag)

        mo_occ = self.mf.mo_occ
        mo_coeff = self.mf.mo_coeff

        P_alpha = self.gen_P(int_r=int_r, mo_coeff=mo_coeff[0], mo_occ=mo_occ[0])
        P_beta = self.gen_P(int_r=int_r, mo_coeff=mo_coeff[1], mo_occ=mo_occ[1])
        P = np.vstack((P_alpha, P_beta))
        return P

    def kernel_TDA(self):
 
        '''for TDA, pure and hybrid share the same form of
                     AX = Xw
            always use the Davidson solver
            pure TDA is not using MZ=Zw^2 form
        '''
        if self.RKS:
            TDA_mv, hdiag = self.gen_RKS_TDA_mv()
            P = self.gen_RKS_P()
        elif self.UKS:
            TDA_mv, hdiag = self.gen_UKS_TDA_mv()
            P = self.gen_UKS_P()
        print('min(hdiag)', min(hdiag)*parameter.Hartree_to_eV)
        energies, X = eigen_solver.Davidson(matrix_vector_product = TDA_mv,
                                                    hdiag = hdiag,
                                                    N_states = self.nroots,
                                                    conv_tol = self.conv_tol,
                                                    max_iter = self.max_iter)
        energies = energies*parameter.Hartree_to_eV
        # print('energies =', energies)

        oscillator_strength = eigen_solver.gen_spectra(energies=energies, 
                                                       transition_vector= X, 
                                                       P=P, 
                                                       name='TDA-ris', 
                                                       RKS=self.RKS)

        return energies, X, oscillator_strength
        # from pyscf.lib import davidson1
        # TDA_vind, _  = self.gen_vind()

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


        # if self.pyscf_TDDFT_vind:
        #     '''
        #     invoke ab-initio TDDFT from PySCF and use our davidson solver
        #     '''
        #     name = 'TDDFT-abinitio'
        #     def TDDFT_mv(X, Y):
        #         '''convert pyscf style (bra) to my style (ket)
        #         return AX + BY and AY + BX'''
        #         XY = np.vstack((X,Y)).T
        #         U = self.pyscf_TDDFT_vind(XY)
        #         A_size = U.shape[1]//2
        #         U1 = U[:,:A_size].T
        #         U2 = -U[:,A_size:].T
        #         return U1, U2
        # else:
        #     name = 'TDDFT-ris'
            
    def kernel_TDDFT(self):     
        if self.a_x != 0:
            '''hybrid TDDFT'''
            if self.RKS:
                P = self.gen_RKS_P()
                TDDFT_hybrid_mv, hdiag = self.gen_RKS_TDDFT_mv()

            elif self.UKS:
                P = self.gen_UKS_P()
                TDDFT_hybrid_mv, hdiag = self.gen_UKS_TDDFT_mv()

            energies, X, Y = eigen_solver.Davidson_Casida(TDDFT_hybrid_mv, hdiag,
                                                            N_states = self.nroots,
                                                            conv_tol = self.conv_tol,
                                                            max_iter = self.max_iter)
        elif self.a_x == 0:
            '''pure TDDFT'''
            if self.RKS:
                TDDFT_pure_mv, hdiag_sq = self.gen_RKS_TDDFT_mv()
                P = self.gen_RKS_P()

            elif self.UKS:
                TDDFT_pure_mv, hdiag_sq = self.gen_UKS_TDDFT_mv()
                P = self.gen_UKS_P()
            # print('min(hdiag_sq)', min(hdiag_sq)**0.5*parameter.Hartree_to_eV)
            energies, Z = eigen_solver.Davidson(TDDFT_pure_mv, hdiag_sq,
                                                N_states = self.nroots,
                                                conv_tol = self.conv_tol,
                                                max_iter = self.max_iter)
            energies = energies**0.5

            X, Y = eigen_solver.XmY_2_XY(Z=Z, AmB_sq=hdiag_sq, omega=energies)

        energies = energies*parameter.Hartree_to_eV
        oscillator_strength = eigen_solver.gen_spectra(energies=energies, 
                                                       transition_vector= X+Y, 
                                                       P=P, 
                                                       name='TDDFT-ris', 
                                                       RKS=self.RKS)
        # print('energies =', energies)
        return energies, X, Y, oscillator_strength
    
     