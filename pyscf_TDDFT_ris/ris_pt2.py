from pyscf import gto, lib, dft
import numpy as np
import multiprocessing as mp
from pyscf_TDDFT_ris import parameter, eigen_solver, math_helper, ris
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=250, threshold=np.inf)

einsum = lib.einsum


class TDDFT_ris_PT2(ris.TDDFT_ris):

    def get_diag_cis(self, mo_energy, n_occ, n_vir):
        vir = mo_energy[n_occ:].reshape(1,n_vir)
        occ = mo_energy[:n_occ].reshape(n_occ,1)
        mo_diff_diag = np.repeat(vir, n_occ, axis=0) - np.repeat(occ, n_vir, axis=1)
        return mo_diff_diag
    
    def kernel(self, method='ris'):
        mf = self.mf
        mo_energy = mf.mo_energy
        mo_diff_diag = self.get_diag_cis(mo_energy=mo_energy,
                                    n_occ=self.n_occ,
                                    n_vir=self.n_vir)
        
        print('mo_diff_diag.shape', mo_diff_diag.shape)
        
        # plt.imshow(mo_diff_diag)
        # plt.colorbar()
        # plt.savefig("mo_diff_diag.pdf")
        # plt.clf()
        # plt.show()

        ''' build the diag(A)^CIS = mo_diff_diag + 2*(ia|ia) - (ii|aa) '''
        a_x = self.a_x
        n_occ = self.n_occ
        n_vir = self.n_vir
        A_size=n_occ*n_vir

        mo_coeff = mf.mo_coeff
        mo_energy = mf.mo_energy

        mol = self.mol

        ''' ris method '''
        if method == 'ris':
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
            
            B_cl_left  = B_ia_cl
            B_cl_right = B_ia_cl
            B_ex_left  = B_ij_ex
            B_ex_right = B_ab_ex


        ''' sTDA method '''
        if method == 'sTDA':

            S = mf.get_ovlp()
            X = math_helper.matrix_power(S, 0.5)
            orthon_c_matrix = np.dot(X,mo_coeff)

            aoslice = mol.aoslice_by_atom()
            N_atm = mol.natm
            N_bf = N_bf = len(mf.mo_occ)
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

            B_cl_left  = einsum("Aia->iaA",q_ia)
            B_cl_right = einsum("Ajb->jbA",GK_q_jb)
            B_ex_left  = einsum("Aij->ijA",q_ij)
            B_ex_right = einsum("Aab->abA",GJ_q_ab)


        A_iajb = einsum('iaA,jbA->iajb',B_cl_left,B_cl_right)

        A_ijab = einsum('ijA,abA->iajb',B_ex_left,B_ex_right)
        # A_ijab = einsum('ijab->iajb', A_ijab)
        
        if method == 'sTDA':
            a_x_tmp = 1
        if method == 'ris':
            a_x_tmp = a_x

        A_matrix = np.diag(mo_diff_diag.reshape(-1,)) + (2*A_iajb - a_x_tmp* A_ijab).reshape(A_size,A_size)
        energy, vec_standard = np.linalg.eigh(A_matrix)
        energy = np.asarray(energy)*parameter.Hartree_to_eV
        print('method =', method)
        print('energy =', energy[:10])

        cl_diag = einsum('iaP,iaP->ia',B_cl_left,B_cl_right)
        ex_diag = einsum('iiP,aaP->ia',B_ex_left,B_ex_right)
        A_diag = mo_diff_diag + 2*cl_diag - a_x_tmp*ex_diag

        # fig, axs = plt.subplots(2,1)
        # vmin = min(mo_diff_diag.min(), A_diag.min())
        # vmax = max(mo_diff_diag.max(), A_diag.max())
        # cax1 = axs[0].imshow(mo_diff_diag, cmap='viridis', vmin=vmin, vmax=vmax)
        # axs[0].set_title('mo_diff_diag')
        # cax2 = axs[1].imshow(A_diag, cmap='viridis', vmin=vmin, vmax=vmax)
        # axs[1].set_title('A_diag')

        # fig.colorbar(cax1, ax=axs.ravel().tolist())
        # # plt.savefig("diag.pdf")
        # plt.show()


        ''' eV to Hartree '''
        sprec_thr = 10/parameter.Hartree_to_eV
        trunc_thr = 2* (1.0 + 0.8 * a_x) * sprec_thr
        print('input spectra window = {:.1f} eV'.format(sprec_thr * parameter.Hartree_to_eV))
        print('input spectra window = {:.3f} a.u.'.format(sprec_thr))
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
        print('number of included occ MO =', n_includ_occ)
        print('number of included vir MO =', n_includ_vir)

        if method == 'ris':
            n_includ_occ = n_occ
            n_includ_vir = n_vir
            sprec_thr = sprec_thr*1.1
        # occ_lumo = A_diag[:,0]
        # homo_vir = A_diag[-1,:]
        # print(occ_lumo)
        # print(homo_vir)
        # n_includ_occ = len(np.where(occ_lumo <= sprec_thr)[0])
        # n_includ_vir = len(np.where(homo_vir <= sprec_thr)[0])
        

        # n_S_CSF =  A_size - n_includ_occ*n_includ_vir


        ''' iterate over all the CSFs to select P-CSF and S-CSF '''

        index_P_CSF_occ = []
        index_P_CSF_vir = []

        index_S_CSF_occ = []
        index_S_CSF_vir = []

        # n_P_CSF = 0
        for io in range(n_occ-n_includ_occ, n_occ):
            for iv in range(n_includ_vir):
                if A_diag[io,iv] <= sprec_thr:
                    # print(io, iv)
                    index_P_CSF_occ.append(io)
                    index_P_CSF_vir.append(iv)
                    # n_P_CSF += 1
                else:
                    index_S_CSF_occ.append(io)
                    index_S_CSF_vir.append(iv)

        print('numnber of P-CSF =', len(index_P_CSF_occ))
        print('numnber of S-CSF to be evaluated by PT2 =', len(index_S_CSF_occ))


        scsf = np.zeros((n_occ, n_vir))
        scsf[n_occ - n_includ_occ:n_occ,:n_includ_vir] = 0.5
        

        index_S_CSF_survive_occ = []
        index_S_CSF_survive_vir = []

        index_S_CSF_sum_occ = []
        index_S_CSF_sum_vir = []

        S_CSF_sum_pt2 = []


        merge_S_CSF = 0
        for i_S in range(len(index_S_CSF_occ)):

            pt2_energy_tmp = np.zeros((len(index_P_CSF_occ),))

            for i_P in range(len(index_P_CSF_occ)):

                i_S_occ = index_S_CSF_occ[i_S]
                i_S_vir = index_S_CSF_vir[i_S]

                i_P_occ = index_P_CSF_occ[i_P]
                i_P_vir = index_P_CSF_vir[i_P]

                scsf[i_P_occ, i_P_vir] = 2.0

                iajb = np.dot(B_cl_left[i_S_occ,i_S_vir,:], B_cl_right[i_P_occ,i_P_vir,:])
                ijab = np.dot(B_ex_left[i_S_occ,i_P_occ,:], B_ex_right[i_S_vir,i_P_vir,:])
    
                A_iajb = 2*iajb - a_x_tmp*ijab

                pt2_energy = A_iajb**2/(A_diag[i_S_occ,i_S_vir] - A_diag[i_P_occ,i_P_vir])
                pt2_energy_tmp[i_P] = pt2_energy
                # total_pt2 += pt2_energy
            total_pt2 = np.sum(pt2_energy_tmp)

            if total_pt2 >= 1e-4:
                merge_S_CSF += 1
                scsf[i_S_occ, i_S_vir] = 1.0
                index_S_CSF_survive_occ.append(i_S_occ)
                index_S_CSF_survive_vir.append(i_S_vir)
            else:
                index_S_CSF_sum_occ.append(i_S_occ)
                index_S_CSF_sum_vir.append(i_S_vir)
                S_CSF_sum_pt2.append(pt2_energy_tmp)

            # print('{:.3e}'.format(total_pt2))

        assert len(index_S_CSF_survive_occ) == merge_S_CSF
        assert len(S_CSF_sum_pt2) == len(index_S_CSF_occ) - merge_S_CSF
        print('number of S-SCF to be merged with P-CSF (can not throw away) =', merge_S_CSF)
        print('number of N-SCF to sum PT2 energy into P-CSF and throw away =', len(S_CSF_sum_pt2))
        n_total_CSF = len(index_P_CSF_occ) + merge_S_CSF
        print('total number of (P+S)-CSF =', n_total_CSF)

        # plt.imshow(scsf, cmap='viridis')
        # plt.show()



        ''' sum PT2 enery into P-SCF '''

        S_CSF_sum_pt2 = np.asarray(S_CSF_sum_pt2)
        total_pt2 = np.sum(S_CSF_sum_pt2, axis=0)
        print('S_CSF_sum_pt2', S_CSF_sum_pt2.shape)
        print('total_pt2', total_pt2.shape)
        print('average PT2 energy lowering: {:.4f} eV'.format(np.average(total_pt2)*parameter.Hartree_to_eV))
        print('maximum PT2 energy lowering: {:.4f} eV'.format(np.max(total_pt2)*parameter.Hartree_to_eV))

        if total_pt2.any():
            for i_P in range(len(index_P_CSF_occ)):

                i_P_occ = index_P_CSF_occ[i_P]
                i_P_vir = index_P_CSF_vir[i_P]

                # mo_diff_diag[i_P_occ, i_P_vir] -= total_pt2[i_P]
                A_diag[i_P_occ, i_P_vir] -= total_pt2[i_P]




        ''' build the A matrix '''
        A_full = np.zeros((n_total_CSF, n_total_CSF))

        index_total_CSF_occ = index_P_CSF_occ + index_S_CSF_survive_occ
        index_total_CSF_vir = index_P_CSF_vir + index_S_CSF_survive_vir

        # for ii in range(len(index_total_CSF_occ)):
        #     for jj in range(len(index_total_CSF_occ)):

        #         i = index_total_CSF_occ[ii]
        #         a = index_total_CSF_vir[ii]

        #         j = index_total_CSF_occ[jj]
        #         b = index_total_CSF_vir[jj]

        #         iajb = np.dot(B_cl_left[i,a,:], B_cl_right[j,b,:])
        #         ijab = np.dot(B_ex_left[i,j,:], B_ex_right[a,b,:])
        
        #         A_full[ii,jj] = 2*iajb - a_x_tmp*ijab

        # for ii in range(len(index_total_CSF_occ)):
        #     i = index_total_CSF_occ[ii]
        #     a = index_total_CSF_vir[ii]
        #     A_full[ii,ii] += mo_diff_diag[i, a]

        assert n_total_CSF == len(index_total_CSF_occ)

        for ii in range(n_total_CSF):

            i = index_total_CSF_occ[ii]
            a = index_total_CSF_vir[ii]

            A_full[ii,ii] = A_diag[i, a]

            for jj in range(ii+1, n_total_CSF):

                j = index_total_CSF_occ[jj]
                b = index_total_CSF_vir[jj]

                iajb = np.dot(B_cl_left[i,a,:], B_cl_right[j,b,:])
                ijab = np.dot(B_ex_left[i,j,:], B_ex_right[a,b,:])
        
                A_full[ii,jj] = 2*iajb - a_x_tmp*ijab
                A_full[jj,ii] = A_full[ii,jj]

        N_states = 5
        energy, vec = np.linalg.eigh(A_full)
        energy = np.asarray(energy)*parameter.Hartree_to_eV
        print('after aggressive truncation')
        # print('method =', method)
        print('energy =', energy[:N_states])
       
        # print(vec[:,:N_states])
        max_value = np.max(np.abs(vec[:,:N_states]))

        # Find the index of the maximum value
        max_index_flat = np.argmax(np.abs(vec[:,:N_states]))
        max_index = np.unravel_index(max_index_flat, vec[:,:N_states].shape)

        print(f"The maximum value is {max_value} and its index is {max_index}")
        print(index_total_CSF_occ)
        print(index_total_CSF_vir)
        X = np.zeros((n_occ, n_vir, N_states))

        for n in range(N_states):
            for jj in range(n_total_CSF):
                i = index_total_CSF_occ[jj]
                a = index_total_CSF_vir[jj]
                X[i, a, n] = vec[jj,n]
        X = X.reshape(n_occ*n_vir, N_states)
        # X = X/np.linalg.norm(X, axis=0)
        # vec_standard[:,:N_states] - X
        X_standard = vec_standard[:,:N_states]
        print('X_standard.shape', X_standard.shape)
        print('X_standard.norm =', np.linalg.norm(X_standard))
        norm =  np.linalg.norm(np.abs(X_standard) -np.abs(X))
        print('X diff =', norm)

        

        





    

