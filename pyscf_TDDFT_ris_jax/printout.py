from pyscf_TDDFT_ris import parameter
import numpy as np

def get_spectra(energies, transition_vector, X_coeff, P, name, RKS, n_occ, n_vir, spectra=True, print_threshold=0.05):
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
    eV, oscillator_strength, cm_1, nm
    '''
    entry = [eV, nm, cm_1, oscillator_strength]
    data = np.zeros((eV.shape[0],len(entry)))
    

    for i in range(4):
        data[:,i] = entry[i]
    print('================================================')
    print('eV       nm       cm^-1    oscillator strength')
    for row in range(data.shape[0]):
        print('{:<8.3f} {:<8.0f} {:<8.0f} {:<8.8f}'.format(data[row,0], data[row,1], data[row,2], data[row,3]))
    
    if spectra:
        filename = name + '_UV_spectra.txt'
        with open(filename, 'w') as f:
            np.savetxt(f, data, fmt='%.8f', header='eV       nm           cm^-1         oscillator strength')
        print('spectra data written to', filename)

    print('================================================')
    


    return oscillator_strength

def print_transition_coeff(X, Y):
    if RKS:
        print('print RKS transition coefficients larger than {:<8f}'.format(print_threshold))
        print('index of HOMO:', n_occ)
        print('index of LUMO:', n_occ+1)
        n_state = X_coeff.shape[1]
        X_coeff = X_coeff.reshape(n_occ,n_vir,n_state)
        for state in range(n_state):
            print('state {:d}:'.format(state+1))
            for occ in range(n_occ):
                for vir in range(n_vir):
                    coeff = X_coeff[occ,vir,state]
                    if np.abs(coeff)>= print_threshold:
                        print('{:>15d}->{:<8d} {:<8.3f}'.format(occ+1, vir+1+n_occ, coeff))
    else:
        print('UKS transition coefficient not printed')