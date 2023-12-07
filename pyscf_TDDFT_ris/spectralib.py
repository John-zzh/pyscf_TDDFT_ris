from pyscf_TDDFT_ris import parameter
import numpy as np

def get_spectra(energies, transition_vector, P, X, Y, name, RKS, n_occ, n_vir,  spectra=True, print_threshold=0.001):
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
    transition_vector is eigenvector of A matrix
    '''
    energies = energies.reshape(-1,)
    # transition_vector = transition_vector/np.linalg.norm(transition_vector, axis=0)
    eV = energies.copy()
    # print(energies, energies.shape)
    cm_1 = eV*8065.544
    nm = 1240.7011/eV

    hartree = energies/parameter.Hartree_to_eV

    trans_dipole_moment = np.dot(P.T, transition_vector)**2

    if RKS:
        trans_dipole_moment *= 2

    '''
    2* because alpha and beta spin
    '''
    oscillator_strength = 2/3 * hartree * np.sum(trans_dipole_moment, axis=0)

    '''
    eV, nm, oscillator_strength
    '''



    
    entry = [eV, nm, cm_1, oscillator_strength]
    data = np.zeros((eV.shape[0],len(entry)))
    for i in range(len(entry)):
        data[:,i] = entry[i]
    print('================================================')
    print('eV       nm       cm^-1    oscillator strength')
    for row in range(data.shape[0]):
        print('{:<8.3f} {:<8.0f} {:<8.0f} {:<8.8f}'.format(data[row,0], data[row,1], data[row,2], data[row,3]))
    


    if spectra:
        filename = name + '_UV_spectra.txt'
        with open(filename, 'w') as f:
            np.savetxt(f, data, fmt='%.5f', header='eV     nm      cm^-1        oscillator_strength')
        print('spectra data written to', filename)

    print('print_threshold:', print_threshold)
    def print_coeff(state, coeff_vec, sybmol, n_occ=n_occ, n_vir=n_vir):
        for occ in range(n_occ):
            for vir in range(n_vir):
                coeff = coeff_vec[occ,vir,state]
                if np.abs(coeff)>= print_threshold:
                    print(f"{occ+1:>15d} {sybmol} {vir+1+n_occ:<8d} {coeff:>15.5f}")

    if RKS:
        print(f"print RKS transition coefficients larger than {print_threshold:.2e}")
        print('index of HOMO:', n_occ)
        print('index of LUMO:', n_occ+1)
        n_state = X.shape[1]
        X = X.reshape(n_occ,n_vir,n_state)
        Y = Y.reshape(n_occ,n_vir,n_state)
        for state in range(n_state):
            print(f" Excited State  {state+1:4d}:      SingletXXXX   \
                 {energies[state]:>.4f} eV  {nm[state]:>.2f} nm  f={oscillator_strength[state]:>.4f}   <S**2>=XXXXX")
            print_coeff(state, X, '->')
            print_coeff(state, Y, '<-')
    else:
        print('UKS transition coefficient not implemenetd yet')

    return oscillator_strength


