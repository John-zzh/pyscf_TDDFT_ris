from pyscf_TDDFT_ris import parameter
import numpy as np



# from pyscf import gto, scf

# # 构建分子
# mol = gto.Mole()
# mol.build(
#     atom='H 0 0 0; F 0 0 1.1',
#     basis='sto-3g'
# )

# # 获取位置和动量积分
# r_integrals = mol.intor('int1e_r', comp=3)  # x, y, z
# p_integrals = mol.intor('int1e_ipovlp', comp=3)  # px, py, pz

# # 磁偶极矩分量 (m_x, m_y, m_z)
# m_x = 0.5j * (r_integrals[1] @ p_integrals[2] - r_integrals[2] @ p_integrals[1])
# m_y = 0.5j * (r_integrals[2] @ p_integrals[0] - r_integrals[0] @ p_integrals[2])
# m_z = 0.5j * (r_integrals[0] @ p_integrals[1] - r_integrals[1] @ p_integrals[0])

# # 打印结果
# print("Magnetic dipole transition moment components:")
# print(f"m_x: {m_x}")
# print(f"m_y: {m_y}")
# print(f"m_z: {m_z}")



def print_coeff(state, coeff_vec, sybmol, n_occ, n_vir, print_threshold):

    abs_coeff = np.abs(coeff_vec[state, :, :])  
    mask = abs_coeff >= print_threshold         

    occ_indices, vir_indices = np.where(mask)

    coeff_values = coeff_vec[state, occ_indices, vir_indices]

    # # 打印结果
    # for occ, vir, coeff in zip(occ_indices, vir_indices, coeff_values):
    #     print(f"{occ+1:>15d} {sybmol} {vir+1+n_occ:<8d} {coeff:>15.5f}")
    results = [ f"{occ+1:>15d} {sybmol} {vir+1+n_occ:<8d} {coeff:>15.5f}" for occ, vir, coeff in zip(occ_indices, vir_indices, coeff_values) ]
    return results

def _charge_center(mol):
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    return np.einsum('z,zr->r', charges, coords)/charges.sum()


def get_spectra(energies, P, X, Y, name, RKS, n_occ, n_vir,  spectra=True, print_threshold=0.001, mdpol=None):
    '''
    E = hν
    c = λ·ν
    E = hc/λ = hck   k in cm-1

    energy in unit eV
    1240.7011/ xyz eV = xyz nm

    oscilator strength f = 2/3 E u


    in general transition dipole moment u =  [Pα]^T [Xα] = Pα^T Xα + Pβ^TXβ + Qα^TYα + Qβ^T Yβ
                                             [Pβ]   [Xβ]
                                             [Qα]   [Yα]
                                             [Qβ]   [Yβ]
    P = Q

    TDA: 
        u =  [Pα]^T [Xα]
             [Pβ]   [Xβ]

        RKS: u = 2 P^T X
        UKS: u = Pα^T Xα + Pβ^TXβ = P^T X (P = [Pα]  X = [Xα])
                                               [Pβ]      [Xβ]
    TDDFT:
        RKS: u = 2 P^T X + 2 P^T Y = 2 P^T(X+Y)
        UKS: u = Pα^T Xα + Pβ^TXβ + Qα^TYα + Qβ^T Yβ =  P^T(X+Y)  (P = [Pα]  X = [Xα] )
                                                                       [Pβ]      [Xβ]
                                                                       [Qα]      [Yα]
                                                                       [Qβ]      [Yβ] 

    for TDA,   f = 2/3 E 2*|<P|X>|**2                   
    for TDDFT, f = 2/3 E 2*|<P|X+Y>|**2     
    P is transition dipole 
    TDA:   transition_vector is X*2**0.5 
    TDDFT: transition_vector is (X*2**0.5 + Y*2**0.5)

    energies are in Hartree
    '''
    energies = energies.reshape(-1,)

    eV = energies.copy() * parameter.Hartree_to_eV
    # print(energies, energies.shape)
    cm_1 = eV*8065.544
    nm = 1240.7011/eV


    if isinstance(Y, np.ndarray):
        trans_dipole_moment = -np.dot(X*2**0.5 + Y*2**0.5, P.T)
    else: 
        trans_dipole_moment = -np.dot(X*2**0.5, P.T)
    print('trans_dipole_moment')
    print(trans_dipole_moment)
    if RKS:

        '''
        2* because alpha and beta spin
        '''
        oscillator_strength = 2/3 * energies * np.sum(2 * trans_dipole_moment**2, axis=1)

    if isinstance(Y, np.ndarray):
        trans_magnetic_moment = -np.dot((X*2**0.5 - Y*2**0.5), mdpol.T )
    else:
        trans_magnetic_moment = -np.dot(X*2**0.5, mdpol.T) 
    print('trans_magnetic_moment')
    print(trans_magnetic_moment)
    rotatory_strength = 500*np.sum(2*trans_dipole_moment * trans_magnetic_moment, axis=1)/2
    print('rotatory_strength:')
    print(rotatory_strength)

    if spectra == True:


        entry = [eV, nm, cm_1, oscillator_strength]
        data = np.zeros((eV.shape[0],len(entry)))
        for i in range(len(entry)):
            data[:,i] = entry[i]
        print('================================================')
        print('eV       nm       cm^-1    oscillator strength')
        for row in range(data.shape[0]):
            print(f'{data[row,0]:<8.3f} {data[row,1]:<8.0f} {data[row,2]:<8.0f} {data[row,3]:<8.8f}')

        filename = name + '_eV_os_Multiwfn.txt'
        with open(filename, 'w') as f:
            np.savetxt(f, data[:,(0,3)], fmt='%.5f', header=f'{len(energies)} 1', comments='')
        print('eV Oscillator strength spectra data written to', filename)

        filename = name + '_eV_rs_Multiwfn.txt'
        with open(filename, 'w') as f:
            new_rs_data = np.hstack((data[:,0].reshape(-1,1), rotatory_strength.reshape(-1,1)))
            np.savetxt(f, new_rs_data, fmt='%.5f', header=f'{len(energies)} 1', comments='')
        print('eV Rotatory strength spectra data written to', filename)


        if RKS:
            print(f"print RKS transition coefficients larger than {print_threshold:.2e}")
            print('index of HOMO:', n_occ)
            print('index of LUMO:', n_occ+1)
            n_state = X.shape[0]
            X = X.reshape(n_state, n_occ, n_vir)
            if isinstance(Y, np.ndarray):
                Y = Y.reshape(n_state, n_occ, n_vir)

            filename = name + '_coeff_Multiwfn.txt'

            with open(filename, 'w') as f:
                
                for state in range(n_state):
                    print(f" Excited State{state+1:4d}:      Singlet-A      {eV[state]:>.4f} eV  {nm[state]:>.2f} nm  f={oscillator_strength[state]:>.4f}   <S**2>=0.000")
                    f.write(f" Excited State{state+1:4d}   1    {eV[state]:>.4f} \n")
                    results = print_coeff(state, X, '->', n_occ=n_occ, n_vir=n_vir, print_threshold=print_threshold)
                    # print(*results, sep='\n')

                    if isinstance(Y, np.ndarray):
                        results += print_coeff(state, Y, '<-', n_occ=n_occ, n_vir=n_vir, print_threshold=print_threshold)

                    print(*results, sep='\n')
                    f.write('\n'.join(results) + '\n\n')
            print('transition coefficient data written to', filename)
        else:
            print('printing UKS transition coefficient not implemented yet')


    return oscillator_strength


