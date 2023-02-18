#
import os, sys
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)
import numpy as np

from TDDFT_ris import parameter, math_helper, diag_ip
from TDDFT_ris.diag_ip import TDDFT_diag_initial_guess, TDDFT_diag_preconditioner
import time



def TDDFT_eigen_solver(matrix_vector_product,
                                    hdiag,
                                    N_states = 20,
                                     conv_tol = 1e-5,
                                     max_iter = 25 ):
    '''
    [ A' B' ] X - [1   0] Y Ω = 0
    [ B' A' ] Y   [0  -1] X   = 0

    A'X = [ diag1   0        0 ] [0]
          [  0   reduced_A   0 ] [X]
          [  0      0     diag3] [0]
    '''


    TD_start = time.time()
    A_size = hdiag.shape[0]

    size_old = 0
    size_new = N_states

    max_N_mv = (max_iter+1)*N_states

    V_holder = np.zeros((A_size, max_N_mv))
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)

    VU1_holder = np.zeros((max_N_mv,max_N_mv))
    VU2_holder = np.zeros_like(VU1_holder)
    WU1_holder = np.zeros_like(VU1_holder)
    WU2_holder = np.zeros_like(VU1_holder)

    VV_holder = np.zeros_like(VU1_holder)
    VW_holder = np.zeros_like(VU1_holder)
    WW_holder = np.zeros_like(VU1_holder)


    '''
    set up initial guess V W, transformed vectors U1 U2
    '''

    (V_holder,
    W_holder,
    size_new,
    energies,
    Xig,
    Yig) = TDDFT_diag_initial_guess(V_holder = V_holder,
                                    W_holder = W_holder,
                                    N_states = size_new,
                                       hdiag = hdiag)

    subcost = 0
    Pcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    full_cost = 0
    for ii in range(max_iter):

        V = V_holder[:,:size_new]
        W = W_holder[:,:size_new]

        '''
        U1 = AV + BW
        U2 = AW + BV
        '''
        # print('size_old =', size_old)
        # print('size_new =', size_new)
        MV_start = time.time()
        U1_holder[:, size_old:size_new], U2_holder[:, size_old:size_new] = matrix_vector_product(
                                                            X=V[:, size_old:size_new],
                                                            Y=W[:, size_old:size_new])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        U1 = U1_holder[:,:size_new]
        U2 = U2_holder[:,:size_new]

        subgenstart = time.time()

        '''
        [U1] = [A B][V]
        [U2]   [B A][W]

        a = [V.T W.T][A B][V] = [V.T W.T][U1] = VU1 + WU2
                     [B A][W]            [U2]
        '''

        (sub_A, sub_B, sigma, pi,
        VU1_holder, WU2_holder, VU2_holder, WU1_holder,
        VV_holder, WW_holder, VW_holder) = math_helper.gen_sub_ab(
                      V_holder, W_holder, U1_holder, U2_holder,
                      VU1_holder, WU2_holder, VU2_holder, WU1_holder,
                      VV_holder, WW_holder, VW_holder,
                      size_old, size_new)

        subgenend = time.time()
        subgencost += subgenend - subgenstart

        '''
        solve the eigenvalue omega in the subspace
        '''
        subcost_start = time.time()
        omega, x, y = math_helper.TDDFT_subspace_eigen_solver(sub_A, sub_B, sigma, pi, N_states)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        '''
        compute the residual
        R_x = U1x + U2y - X_full*omega
        R_y = U2x + U1y + Y_full*omega
        X_full = Vx + Wy
        Y_full = Wx + Vy
        '''
        full_cost_start = time.time()
        X_full = np.dot(V,x)
        X_full += np.dot(W,y)

        Y_full = np.dot(W,x)
        Y_full += np.dot(V,y)

        R_x = np.dot(U1,x)
        R_x += np.dot(U2,y)
        R_x -= X_full*omega

        R_y = np.dot(U2,x)
        R_y += np.dot(U1,y)
        R_y += Y_full*omega

        full_cost_end = time.time()
        full_cost += full_cost_end - full_cost_start

        residual = np.vstack((R_x, R_y))
        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)
        print('step ', ii+1, 'max_norm =', max_norm)
        if max_norm < conv_tol or ii == (max_iter -1):
            break

        index = [r_norms.index(i) for i in r_norms if i > conv_tol]

        '''
        preconditioning step
        '''
        X_new, Y_new = TDDFT_diag_preconditioner(R_x = R_x[:,index],
                                                   R_y = R_y[:,index],
                                                 omega = omega[index],
                                                 hdiag = hdiag)

        '''
        GS and symmetric orthonormalization
        '''
        size_old = size_new
        GScost_start = time.time()
        V_holder, W_holder, size_new = math_helper.VW_Gram_Schmidt_fill_holder(
                                            V_holder = V_holder,
                                            W_holder = W_holder,
                                               X_new = X_new,
                                               Y_new = Y_new,
                                                   m = size_old,
                                              double = False)
        GScost_end = time.time()
        GScost += GScost_end - GScost_start

        if size_new == size_old:
            print('All new guesses kicked out during GS orthonormalization')
            break

    TD_end = time.time()

    TD_cost = TD_end - TD_start

    if ii == (max_iter -1):
        print('=== TDDFT eigen solver Failed Due to Iteration Limit ===')
        print('current residual norms', r_norms)
    else:
        print('TDDFT eigen solver Guess Done' )

    print('Finished in {:d} steps, {:.2f} seconds'.format(ii+1, TD_cost))
    print('final subspace', sub_A.shape[0])
    print('max_norm = {:.2e}'.format(max_norm))
    for enrty in ['MVcost','GScost','subgencost','subcost','full_cost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s {:<5.2%}".format(enrty, cost, cost/TD_cost))

    energies = omega*parameter.Hartree_to_eV

    return energies, X_full, Y_full

def gen_spectra(energies, transition_vector, P, name):
    '''
    E = hν
    c = λ·ν
    E = hc/λ = hck   k in cm-1

    energy in unit eV
    1240.7011/ xyz eV = xyz nm

    for TDA,   f = 2/3 E |<P|X>|**2     ???
    for TDDFT, f = 2/3 E |<P|X+Y>|

    '''
    energies = energies.reshape(-1,)

    eV = energies.copy()
    # print(energies, energies.shape)
    cm_1 = eV*8065.544
    nm = 1240.7011/eV

    '''
    P is right-hand-side of polarizability
    transition_vector is eigenvector of A matrix
    '''

    hartree = energies/parameter.Hartree_to_eV
    trans_dipole = np.dot(P.T, transition_vector)

    trans_dipole = 2*trans_dipole**2
    '''
    2* because alpha and beta spin
    '''
    oscillator_strength = 2/3 * hartree * np.sum(trans_dipole, axis=0)

    '''
    eV, oscillator_strength, cm_1, nm
    '''
    entry = [eV, nm, cm_1, oscillator_strength]
    data = np.zeros((eV.shape[0],len(entry)))
    for i in range(4):
        data[:,i] = entry[i]

    filename = name + '_UV_spectra.txt'
    with open(filename, 'w') as f:
        np.savetxt(f, data, fmt='%.8f', header='eV       nm           cm^-1         oscillator strength')
    print('spectra written to', filename, '\n')
