from pyscf import gto, lib, dft
import numpy as np
import concurrent.futures
from pyscf_TDDFT_ris import parameter, eigen_solver, math_helper, TDDFT_ris, spectralib, eigen_solver
import matplotlib.pyplot as plt
import time

np.set_printoptions(linewidth=250, threshold=np.inf)

einsum = lib.einsum


def make_progress_dict(list_length):
    progress_dict = {}
    progress_dict[int(list_length*0.25) - 1] = '25% progress reached'
    progress_dict[int(list_length*0.50) - 1] = '50% progress reached'
    progress_dict[int(list_length*0.75) - 1] = '75% progress reached'
    progress_dict[int(list_length*1.00) - 1] = '100% progress reached'
    return progress_dict

def process_iteration_all_S_pt2_against_i_P(i_P):

    if i_P in progress_P_CSF_dict:
        print(progress_P_CSF_dict[i_P])
    
    i_P_occ = P_CSF_indices[0][i_P]
    i_P_vir = P_CSF_indices[1][i_P]

    iajb_vec = einsum('SQ,Q->S',B_cl_left[S_CSF_indices[0],S_CSF_indices[1],:], B_cl_right[i_P_occ,i_P_vir,:])
    # print('iajb_vec.shape', iajb_vec.shape)
    ijab_vec = einsum('SQ,SQ->S',B_ex_left[S_CSF_indices[0],i_P_occ,:], B_ex_right[S_CSF_indices[1],i_P_vir,:])
    pt2_energy_vec = 2*iajb_vec - a_x_tmp*ijab_vec
    pt2_energy_vec = pt2_energy_vec**2/(A_diag[S_CSF_indices[0],S_CSF_indices[1]] - A_diag[i_P_occ,i_P_vir])
    # print('pt2_energy_vec.shape', pt2_energy_vec.shape)

    return pt2_energy_vec


def process_iteration_get_ijab(i_CSF):
    '''(ij|ab) = Σ_Q  B_ij^Q B_ab^Q
    '''
    if i_CSF in progress_final_CSF_dict:
        print(progress_final_CSF_dict[i_CSF])

    i = final_CSF_indices[0][i_CSF]
    a = final_CSF_indices[1][i_CSF]

    B_ex_left_selected  =  B_ex_left[i,final_CSF_indices[0],:]
    B_ex_right_selected = B_ex_right[a,final_CSF_indices[1],:]
    ijab = einsum('mQ,mQ->m',B_ex_left_selected, B_ex_right_selected)

    ijab = ijab.reshape(-1,)

    return ijab

def process_iteration_get_ibja(i_CSF):
    '''(ij|ab) = Σ_Q  B_ij^Q B_ab^Q
    '''
    if i_CSF in progress_final_CSF_dict:
        print(progress_final_CSF_dict[i_CSF])

    i = final_CSF_indices[0][i_CSF]
    a = final_CSF_indices[1][i_CSF]

    B_cl_left_selected  =  B_cl_left[i,final_CSF_indices[1],:]
    B_cl_right_selected = B_cl_right[final_CSF_indices[0],a,:]
    ibja = einsum('mQ,mQ->m',B_cl_left_selected, B_cl_right_selected)

    return ibja 


class TDDFT_ris_PT2(TDDFT_ris.TDDFT_ris):

    def __init__(self, parallel=True, spectra_window=10, pt2_tol=1e-4, method='ris', N_cpus=4, truncMO=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.parallel = parallel
        self.spectra_window = spectra_window
        self.pt2_tol = pt2_tol
        self.N_cpus = N_cpus
        self.truncMO = truncMO
        print(f"parallel with {self.N_cpus} CPUs ")

    def get_B_cl_ex_matrix(self):

        mo_energy = self.mf.mo_energy

        n_occ = self.n_occ
        n_vir = self.n_vir
        vir = mo_energy[n_occ:].reshape(1,n_vir)
        occ = mo_energy[:n_occ].reshape(n_occ,1)
        mo_diff_diag = np.repeat(vir, n_occ, axis=0) - np.repeat(occ, n_vir, axis=1)
        mo_diff_diag = mo_diff_diag.astype(np.float32)

        ''' build the diag(A)^CIS = mo_diff_diag + 2*(ia|ia) - (ii|aa) '''
        a_x = self.a_x
        n_occ = self.n_occ
        n_vir = self.n_vir
        A_size = n_occ*n_vir

        mo_coeff = self.mf.mo_coeff


        mol = self.mol

        ''' ris method '''
        if self.method == 'ris':
            auxmol = self.gen_auxmol(theta=0.2, add_p=False)
            eri2c, eri3c = self.gen_eri2c_eri3c(mol=self.mol, auxmol=auxmol, omega=0)
            uvQL = self.gen_uvQL(eri2c=eri2c, eri3c=eri3c)

            '''hybrid RKS TDA'''
            B_ia_cl, _, B_ij_ex, B_ab_ex = self.gen_B_cl_B_ex(mol=mol,
                                                                auxmol=auxmol,
                                                                uvQL=uvQL, 
                                                                eri3c=eri3c, 
                                                                eri2c=eri2c,
                                                                n_occ=n_occ, 
                                                                mo_coeff=mo_coeff)
            
            B_cl_left  = B_ia_cl.astype(np.float32)
            B_cl_right = B_ia_cl.astype(np.float32)
            B_ex_left  = B_ij_ex.astype(np.float32)
            B_ex_right = B_ab_ex.astype(np.float32)

        ''' sTDA method '''
        if self.method == 'sTDA':

            S = self.mf.get_ovlp()
            X = math_helper.matrix_power(S, 0.5)
            orthon_c_matrix = np.dot(X,mo_coeff)

            aoslice = mol.aoslice_by_atom()
            N_atm = mol.natm
            N_bf = N_bf = len(self.mf.mo_occ)
            q_tensors = np.zeros([N_atm, N_bf, N_bf])
            for atom_id in range(N_atm):
                shst, shend, atstart, atend = aoslice[atom_id]
                q_tensors[atom_id,:,:] = np.dot(orthon_c_matrix[atstart:atend,:].T,
                                                orthon_c_matrix[atstart:atend,:])
            q_ij = np.zeros((N_atm, n_occ, n_occ))
            q_ij[:,:,:] = q_tensors[:,:n_occ,:n_occ]

            q_ab = np.zeros((N_atm, n_vir, n_vir))
            q_ab[:,:,:] = q_tensors[:,n_occ:,n_occ:]

            q_ia = np.zeros((N_atm, n_occ, n_vir))
            print('q_ia.shape', q_ia.shape)
            q_ia[:,:,:] = q_tensors[:,:n_occ,n_occ:]

            ''' Gamma J&K '''
            eta = [parameter.HARDNESS[mol.atom_pure_symbol(atom_id)] for atom_id in range(N_atm)]
            eta = np.asarray(eta).reshape(1,-1)
            eta = (eta + eta.T)/2
            R_array = gto.mole.inter_distance(mol, coords=None)

            alpha, beta = parameter.gen_sTDA_alpha_beta_ax(a_x)
            print('alpha, beta =', alpha, beta )
            GammaK = (R_array**alpha + eta**(-alpha)) **(-1/alpha)
            GammaJ = (R_array**beta + (a_x * eta)**(-beta))**(-1/beta)

            GK_q_jb = einsum("Bjb,AB->Ajb", q_ia, GammaK)
            GJ_q_ab = einsum("Bab,AB->Aab", q_ab, GammaJ)

            B_cl_left  = einsum("Aia->iaA",q_ia).astype(np.float32)
            B_cl_right = einsum("Ajb->jbA",GK_q_jb).astype(np.float32)
            B_ex_left  = einsum("Aij->ijA",q_ij).astype(np.float32)
            B_ex_right = einsum("Aab->abA",GJ_q_ab).astype(np.float32)

        if self.method == 'sTDA':
            a_x_tmp = 1
        if self.method == 'ris':
            a_x_tmp = a_x

        # A_iajb = einsum('iaA,jbA->iajb',B_cl_left,B_cl_right)
        # A_ijab = einsum('ijA,abA->iajb',B_ex_left,B_ex_right)
        # A_ijab = einsum('ijab->iajb', A_ijab)
        
        # A_matrix = np.diag(mo_diff_diag.reshape(-1,)) + (2*A_iajb - a_x_tmp* A_ijab).reshape(A_size,A_size)
        # energy, vec_standard = np.linalg.eigh(A_matrix)
        # energy = np.asarray(energy)*parameter.Hartree_to_eV
        # print('method =', method)
        # print('energy =', energy[:10])

        cl_diag = einsum('iaP,iaP->ia',B_cl_left,B_cl_right)
        ex_diag = einsum('iiP,aaP->ia',B_ex_left,B_ex_right)
        A_diag = mo_diff_diag + 2*cl_diag - a_x_tmp*ex_diag

        return B_cl_left, B_cl_right, B_ex_left, B_ex_right, mo_diff_diag, a_x_tmp, A_diag

    def get_n_included_MO(self, spectra_window=10):
        ''' eV to Hartree '''
        spec_thr = spectra_window/parameter.Hartree_to_eV

        n_occ = self.n_occ
        n_vir = self.n_vir

        if self.truncMO == True:
            mo_energy = self.mf.mo_energy

            a_x = self.a_x
            
            if self.method == 'ris':
                fold = 2
            if self.method == 'sTDA':
                fold = 2

            trunc_thr = fold* (1.0 + 0.8 * a_x) * spec_thr
            print('input spectra window = {:.1f} eV'.format(spec_thr * parameter.Hartree_to_eV))
            print('input spectra window = {:.3f} a.u.'.format(spec_thr))
            print('truncation threshold = {:.1f} eV'.format(trunc_thr * parameter.Hartree_to_eV))

            ''' number of included occupied/virtual MOs after truncation '''
            E_homo = mo_energy[n_occ-1]
            E_lumo = mo_energy[n_occ]

            print('E_homo = {:.3f} eV'.format(E_homo* parameter.Hartree_to_eV))
            print('E_lumo = {:.3f} eV'.format(E_lumo* parameter.Hartree_to_eV))
        
            occ_cut_off = E_lumo - trunc_thr
            vir_cut_off = E_homo + trunc_thr
            print('occ MO cut-off: {:.3f} eV'.format(occ_cut_off * parameter.Hartree_to_eV))
            print('vir MO cut-off: {:.3f} eV'.format(vir_cut_off * parameter.Hartree_to_eV))
            
            n_includ_occ = len(np.where(mo_energy[:n_occ] > occ_cut_off)[0])
            n_includ_vir = len(np.where(mo_energy[n_occ:] < vir_cut_off)[0])

            n_includ_occ = int(n_occ*1)
            n_includ_vir = int(n_vir*1)
        else:
            n_includ_occ = n_occ
            n_includ_vir = n_vir
            spec_thr = spec_thr*1.1
        # occ_lumo = A_diag[:,0]
        # homo_vir = A_diag[-1,:]
        # print(occ_lumo)
        # print(homo_vir)
        print('number of included occ MO =', n_includ_occ)
        print('number of included vir MO =', n_includ_vir)
        return n_includ_occ, n_includ_vir, spec_thr

    def get_P_S_CSF_indices_pairs(self, A_diag, spec_thr, n_includ_occ, n_includ_vir):

        n_occ = self.n_occ
        n_vir = self.n_vir

        if n_includ_occ == n_occ and n_includ_vir == n_vir:
            print('no MO truncation')
            P_CSF_indices = np.where(A_diag[:,:] < spec_thr)
            S_CSF_indices = np.where(A_diag[:,:] >= spec_thr)
        else:
            print('A_diag[n_occ-n_includ_occ:,:n_includ_vir].shape', A_diag[n_occ-n_includ_occ:,:n_includ_vir].shape)
            P_CSF_indices = np.where(A_diag[n_occ-n_includ_occ:,:n_includ_vir] < spec_thr)
            S_CSF_indices = np.where(A_diag[n_occ-n_includ_occ:,:n_includ_vir] >= spec_thr)
            P_CSF_indices = (P_CSF_indices[0] + n_occ-n_includ_occ, P_CSF_indices[1])
            


        print('-------------------------------------------------')
        print('numnber of P-CSF =', len(P_CSF_indices[0]))
        print('numnber of S-CSF =', len(S_CSF_indices[1]))
        
        return P_CSF_indices, S_CSF_indices


    def get_final_CSF_indices_parallel(self, pt2_tol=1e-4):
        
        global progress_P_CSF_dict
        progress_P_CSF_dict = make_progress_dict(len(P_CSF_indices[0]))

        print('parallized looping over all P_CSF：')
        args = range(len(P_CSF_indices[0]))
        # with concurrent.futures.ProcessPoolExecutor(max_workers=self.N_cpus) as executor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.N_cpus) as executor:
            results = executor.map(process_iteration_all_S_pt2_against_i_P, args)
        print('parallized looping over all P_CSF done')

        # size (P,S)
        pt2_energy_array = np.asarray(list(results))

        # size (S)
        total_pt2_each_S_against_all_P = np.sum(pt2_energy_array, axis=0)

        survive = np.where(total_pt2_each_S_against_all_P >= pt2_tol)
        neglect = np.where(total_pt2_each_S_against_all_P < pt2_tol)

        print('number of survived  S_CSF:', survive[0].shape[0])
        print('number of neglected S_CSF:', neglect[0].shape[0])

        S_CSF_survive_indices = (S_CSF_indices[0][survive[0]], S_CSF_indices[1][survive[0]])
        # print('S_CSF_survive_indices', S_CSF_survive_indices)
        S_CSF_neglect_pt2_energy = np.sum(pt2_energy_array[:, neglect[0]], axis=1)
        final_CSF_indices = (np.hstack((P_CSF_indices[0],S_CSF_survive_indices[0])),  np.hstack((P_CSF_indices[1], S_CSF_survive_indices[1])))
        # print('S_CSF_neglect_pt2_energy.shape', S_CSF_neglect_pt2_energy.shape)

        return final_CSF_indices, S_CSF_neglect_pt2_energy



    def get_diag_iajb_ijab(self, S_CSF_neglect_pt2_energy):
        ''' build '''

        n_final_CSF = len(final_CSF_indices[1])

        print('final number of CSF =', n_final_CSF)

        iajb = einsum('mQ,nQ->mn',B_cl_left[final_CSF_indices[0],final_CSF_indices[1],:], B_cl_right[final_CSF_indices[0],final_CSF_indices[1],:])
        # print('check iajb norm', np.linalg.norm(iajb_full-iajb))

        global progress_final_CSF_dict
        progress_final_CSF_dict = make_progress_dict(len(final_CSF_indices[0]))
        print('Building ijab')
        print('parallized looping over all final_CSF：')
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.N_cpus) as executor:
            results = executor.map(process_iteration_get_ijab, range(len(final_CSF_indices[0])))
        ijab = np.asarray(list(results))
        print('parallized looping over all final_CSF done')

        mo_diff_diag[P_CSF_indices[0], P_CSF_indices[1]] -= S_CSF_neglect_pt2_energy
        mo_diff_diag_final = mo_diff_diag[final_CSF_indices[0], final_CSF_indices[1]]

        mo_diff_diag_final = np.diag(mo_diff_diag_final.reshape(-1,))

        # print('check iajb norm', np.linalg.norm(iajb_full-iajb))
        # print('check ijab norm', np.linalg.norm(ijab_full-ijab))

        return mo_diff_diag_final, iajb, ijab


    def eigen_A_matrix(self, mo_diff_diag_final, iajb, ijab):

        A_full = mo_diff_diag_final + 2*iajb - a_x_tmp*ijab

        N_states = self.nroots
        start = time.time()
        # energies, vec = np.linalg.eigh(A_full)
        
        def TDA_mv(vec):
            return np.dot(A_full, vec)

        energies, vec = eigen_solver.Davidson(matrix_vector_product = TDA_mv,
                                                    hdiag = np.diag(A_full),
                                                    N_states = N_states,
                                                    conv_tol = self.conv_tol,
                                                    max_iter = self.max_iter)
        end = time.time()

        print(f"time for eigen solver = {end - start:.0f} seconds")

        energies = np.asarray(energies[:N_states])*parameter.Hartree_to_eV

        print('energies =', energies)
       
        n_occ = self.n_occ
        n_vir = self.n_vir
        X = np.zeros((n_occ, n_vir, N_states))

        X[final_CSF_indices[0], final_CSF_indices[1], :] = vec[:,:N_states]

        X = X.reshape(n_occ*n_vir, N_states)

        P = self.gen_RKS_P()
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



    def eigen_AB_matrix(self, mo_diff_diag_final, iajb, ijab):
        A_full = mo_diff_diag_final + 2*iajb - a_x_tmp*ijab

        start = time.time()
        print('Building ibja')
        print('parallized looping over all final_CSF：')
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.N_cpus) as executor:
            results = executor.map(process_iteration_get_ibja, range(len(final_CSF_indices[0])))
        ibja = np.asarray(list(results))
        end = time.time()
        print('parallized looping over all final_CSF done')
        print(f"time for buld ibja           : {end - start:5.0f} seconds")


        B_full = 2*iajb - a_x_tmp*ibja

        def TDDFT_mv(X, Y):
            AX_p_BY = np.dot(A_full, X) +  np.dot(B_full, Y)
            BX_p_AY = np.dot(B_full, X) +  np.dot(A_full, Y)
            return AX_p_BY, BX_p_AY

        N_states = self.nroots
        energies, X_vec, Y_vec = eigen_solver.Davidson_Casida(matrix_vector_product = TDDFT_mv, 
                                                        hdiag=np.diag(A_full),
                                                        N_states = N_states,
                                                        conv_tol = self.conv_tol,
                                                        max_iter = self.max_iter)

        energies = np.asarray(energies[:N_states])*parameter.Hartree_to_eV
        print('energies =', energies)

        n_occ = self.n_occ
        n_vir = self.n_vir
        X = np.zeros((n_occ, n_vir, N_states))
        Y = np.zeros((n_occ, n_vir, N_states))

        X[final_CSF_indices[0], final_CSF_indices[1], :] = X_vec[:,:N_states]
        Y[final_CSF_indices[0], final_CSF_indices[1], :] = Y_vec[:,:N_states]
        X = X.reshape(n_occ*n_vir, N_states)
        Y = Y.reshape(n_occ*n_vir, N_states)

        P = self.gen_RKS_P()
        oscillator_strength = spectralib.get_spectra(energies=energies, 
                                                       transition_vector= X+Y, 
                                                       X=X/(2**0.5),
                                                       Y=Y/(2**0.5),
                                                       P=P, 
                                                       name=self.out_name+'_TDDFT_ris', 
                                                       RKS=self.RKS,
                                                       spectra=self.spectra,
                                                       print_threshold = self.print_threshold,
                                                       n_occ=self.n_occ if self.RKS else (self.n_occ_a, self.n_occ_b),
                                                       n_vir=self.n_vir if self.RKS else (self.n_vir_a, self.n_vir_b))

        return energies, X, Y, oscillator_strength


    def kernel_prepare(self):
        start = time.time()
        global B_cl_left, B_cl_right, B_ex_left, B_ex_right, mo_diff_diag, a_x_tmp, A_diag
        B_cl_left, B_cl_right, B_ex_left, B_ex_right, mo_diff_diag, a_x_tmp, A_diag = self.get_B_cl_ex_matrix()

        n_includ_occ, n_includ_vir, spec_thr = self.get_n_included_MO(spectra_window=self.spectra_window)


        global P_CSF_indices, S_CSF_indices
        P_CSF_indices, S_CSF_indices = self.get_P_S_CSF_indices_pairs(A_diag=A_diag, 
                                                                    spec_thr=spec_thr, 
                                                                    n_includ_occ=n_includ_occ, 
                                                                    n_includ_vir=n_includ_vir)
        
        global final_CSF_indices
        end0 = time.time()
        final_CSF_indices, S_CSF_neglect_pt2_energy = self.get_final_CSF_indices_parallel(pt2_tol=self.pt2_tol)
        end1 =  time.time()

        mo_diff_diag_final, iajb, ijab = self.get_diag_iajb_ijab(S_CSF_neglect_pt2_energy=S_CSF_neglect_pt2_energy)
        end2 =  time.time()
        print(f"time for build rank-3 tensors : {end0 - start:5.0f} seconds")
        print(f"time for compute PT2 energy   : {end1 - end0:5.0f} seconds")  
        print(f"time for build iajb, ijab     : {end2 - end1:5.0f} seconds")
        print(f"parallelized with             : {self.N_cpus:5d} CPUs")

        return mo_diff_diag_final, iajb, ijab
  

    def kernel_TDA(self):
        mo_diff_diag_final, iajb, ijab = self.kernel_prepare()

        energy, X, oscillator_strength = self.eigen_A_matrix(mo_diff_diag_final=mo_diff_diag_final, iajb=iajb, ijab=ijab)    

        return energy, X, oscillator_strength

    def kernel_TDDFT(self):
        mo_diff_diag_final, iajb, ijab = self.kernel_prepare()

        energy, X, Y, oscillator_strength = self.eigen_AB_matrix(mo_diff_diag_final=mo_diff_diag_final, iajb=iajb, ijab=ijab)    

        return energy, X, Y, oscillator_strength     





    


    # def get_final_CSF_indices_vectorize(self, B_cl_left, B_cl_right, B_ex_left, B_ex_right, a_x_tmp, A_diag, P_CSF_indices, S_CSF_indices, P_CSF_indices_pairs, S_CSF_indices_pairs, pt2_tol=1e-4):
    #     ''' vectorize the calculation of PT2 energy 
    #         too much memory!!!!!
    #     '''
        

    #     # size (S, N_auxbf)
    #     B_cl_left_S_P  = np.array([ B_cl_left[i,a,:] for i,a in S_CSF_indices_pairs])
    #     print('B_cl_left_S_P.shape', B_cl_left_S_P.shape)
        
    #     # size (P, N_auxbf)
    #     B_cl_right_S_P = np.array([B_cl_right[i,a,:] for i,a in P_CSF_indices_pairs])
    #     print('B_cl_right_S_P.shape', B_cl_right_S_P.shape)

    #     # size (S, P)
    #     iajb_S_P = einsum('SQ,PQ->SP', B_cl_left_S_P, B_cl_right_S_P)
    #     print('iajb_S_P.shape', iajb_S_P.shape)

    #     # size (S,P, N_auxbf)
    #     S_indices = [i for i, _ in S_CSF_indices_pairs]
    #     P_indices = [j for j, _ in P_CSF_indices_pairs]
    #     B_ex_left_S_P = B_ex_left[np.ix_(S_indices, P_indices)].reshape(len(S_CSF_indices_pairs), len(P_CSF_indices_pairs), -1)
    #     # B_ex_left_S_P  = np.array([ B_ex_left[i,j,:] for i,_ in S_CSF_indices_pairs for j,_ in P_CSF_indices_pairs]).reshape(len(S_CSF_indices_pairs), len(P_CSF_indices_pairs), -1)
    #     print('B_ex_left_S_P.shape', B_ex_left_S_P.shape)

    #     # size (S,P, N_auxbf)
    #     S_indices = [a for _, a in S_CSF_indices_pairs]
    #     P_indices = [b for _, b in P_CSF_indices_pairs]
    #     B_ex_right_S_P = B_ex_right[np.ix_(S_indices, P_indices)].reshape(len(S_CSF_indices_pairs), len(P_CSF_indices_pairs), -1)
    #     # B_ex_right_S_P = np.array([B_ex_right[a,b,:] for _,a in S_CSF_indices_pairs for _,b in P_CSF_indices_pairs]).reshape(len(S_CSF_indices_pairs), len(P_CSF_indices_pairs), -1)
    #     print('B_ex_right_S_P.shape', B_ex_right_S_P.shape)

    #     # size (S, P)
    #     ijab_S_P = einsum('SPQ,SPQ->SP', B_ex_left_S_P, B_ex_right_S_P)
    #     print('ijab_S_P.shape', ijab_S_P.shape)


    #     # size (S,P)
    #     A_iajb_S_P = 2*iajb_S_P - a_x_tmp*ijab_S_P
    #     print('A_iajb_S_P.shape', A_iajb_S_P.shape)

    #     A_diag_S = np.array([A_diag[i,a] for i, a in S_CSF_indices_pairs])
    #     A_diag_P = np.array([A_diag[i,a] for i, a in P_CSF_indices_pairs])

    #     # size (S,P)
    #     A_diag_diff_S_P = np.repeat(A_diag_S.reshape(-1,1), len(P_CSF_indices_pairs), axis=1) - np.repeat(A_diag_P.reshape(1,-1), len(S_CSF_indices_pairs), axis=0)
    #     print('A_diag_diff_S_P.shape', A_diag_diff_S_P.shape)

    #     # size (S,P)
    #     pt2_energy_S_P = A_iajb_S_P**2/A_diag_diff_S_P 
    #     print('pt2_energy_S_P.shape', pt2_energy_S_P.shape)

    #     # size (S,)
    #     pt2_energy_S_P_sum = np.sum(pt2_energy_S_P, axis=1)
    #     print('pt2_energy_S_P_sum.shape', pt2_energy_S_P_sum.shape)

        

    #     S_CSF_survive_indices = np.where(pt2_energy_S_P_sum > pt2_tol)
    #     S_CSF_neglect_indices = np.where(pt2_energy_S_P_sum <= pt2_tol)
    #     # print('S_CSF_survive_indices', S_CSF_survive_indices)
    #     print('number of survived  S_CSF:', S_CSF_survive_indices[0].shape[0])
    #     print('number of neglected S_CSF:', S_CSF_neglect_indices[0].shape[0])
        
    #     S_CSF_survive_pairs = [S_CSF_indices_pairs[i] for i in S_CSF_survive_indices[0]] 
    #     # S_CSF_neglect_pairs = [S_CSF_indices_pairs[i] for i in S_CSF_neglect_indices[0]] 

    #     # size ( neglected_S_CSF, P)
    #     S_CSF_sum_pt2 = pt2_energy_S_P[S_CSF_neglect_indices[0],:]
    #     # print('S_CSF_sum_pt2.shape', S_CSF_sum_pt2.shape)

    #     return S_CSF_survive_pairs, S_CSF_sum_pt2


    # def get_final_CSF_indices_iterative(self, B_cl_left, B_cl_right, B_ex_left, B_ex_right, a_x_tmp, A_diag, P_CSF_indices_pairs, S_CSF_indices_pairs, pt2_tol=1e-4):

    #     S_CSF_survive_pairs = []
    #     # index_S_CSF_sum_pairs = []
    #     S_CSF_sum_pt2 = []
    #     # survive_S_CSF = 0

    #     for i_S in range(len(S_CSF_indices_pairs)):
    #         if i_S % 100 == 0:
    #             print('i_S =', i_S)
    #         pt2_energy_tmp = np.zeros((len(P_CSF_indices_pairs),))
    #         i_S_occ, i_S_vir = S_CSF_indices_pairs[i_S]

    #         for i_P in range(len(P_CSF_indices_pairs)):

    #             i_P_occ, i_P_vir = P_CSF_indices_pairs[i_P]

    #             # scsf[i_P_occ, i_P_vir] = 2.0

    #             iajb = np.dot(B_cl_left[i_S_occ,i_S_vir,:], B_cl_right[i_P_occ,i_P_vir,:])
    #             ijab = np.dot(B_ex_left[i_S_occ,i_P_occ,:], B_ex_right[i_S_vir,i_P_vir,:])
    
    #             A_iajb = 2*iajb - a_x_tmp*ijab

    #             pt2_energy = A_iajb**2/(A_diag[i_S_occ,i_S_vir] - A_diag[i_P_occ,i_P_vir])
    #             pt2_energy_tmp[i_P] = pt2_energy
    #             # total_pt2 += pt2_energy
    #         total_pt2 = np.sum(pt2_energy_tmp)

    #         if total_pt2 >= pt2_tol:
    #             # survive
    #             S_CSF_survive_pairs.append((i_S_occ, i_S_vir))

    #         else:
    #             # neglect
    #             S_CSF_sum_pt2.append(pt2_energy_tmp)

    #         # print('{:.3e}'.format(total_pt2))

    #     # assert len(S_CSF_indices_pairs) == survive_S_CSF
    #     # assert len(S_CSF_sum_pt2) == len(S_CSF_indices_pairs) - survive_S_CSF
    #     print('number of S-SCF to be merged with P-CSF (can not throw away) =', len(S_CSF_survive_pairs))
    #     print('number of N-SCF to sum PT2 energy into P-CSF and throw away =', len(S_CSF_sum_pt2))
    #     # n_final_CSF = len(P_CSF_indices_pairs) + survive_S_CSF
    #     # print('total number of (P+S)-CSF =', n_final_CSF)

    #     S_CSF_sum_pt2 = np.asarray(S_CSF_sum_pt2)
    #     print('S_CSF_sum_pt2.shape', S_CSF_sum_pt2.shape)
    #     return S_CSF_survive_pairs, S_CSF_sum_pt2


        # A_full = np.zeros((n_final_CSF, n_final_CSF))
        # for ii in range(n_final_CSF):

        #     i = final_CSF_indices[0][ii]
        #     a = final_CSF_indices[1][ii]

        #     A_full[ii,ii] = A_diag[i, a]

        #     for jj in range(ii+1, n_final_CSF):

        #         j = final_CSF_indices[0][jj]
        #         b = final_CSF_indices[1][jj]
        #         iajb = np.dot(B_cl_left[i,a,:], B_cl_right[j,b,:])
        #         ijab = np.dot(B_ex_left[i,j,:], B_ex_right[a,b,:])
        #          
        
        #         A_full[ii,jj] = 2*iajb - a_x_tmp*ijab
        #         A_full[jj,ii] = A_full[ii,jj]

        # iajb_full = np.zeros((n_final_CSF, n_final_CSF))
        # ijab_full = np.zeros((n_final_CSF, n_final_CSF))
        # for ii in range(n_final_CSF):
                
        #         i = final_CSF_indices[0][ii]
        #         a = final_CSF_indices[1][ii]
    
        #         for jj in range(n_final_CSF):
    
        #             j = final_CSF_indices[0][jj]
        #             b = final_CSF_indices[1][jj]

        #             iajb = np.dot(B_cl_left[i,a,:], B_cl_right[j,b,:])
        #             ijab = np.dot(B_ex_left[i,j,:], B_ex_right[a,b,:])

        #             iajb_full[ii,jj] = iajb
        #             iajb_full[jj,ii] = iajb
          
        #             ijab_full[ii,jj] = ijab
        #             ijab_full[jj,ii] = ijab