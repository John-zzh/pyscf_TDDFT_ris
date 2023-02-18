# -*- coding: utf-8 -*-

import os,sys
'''
absolute import
'''
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

from TDDFT_ris import parameter
import numpy as np

def TDA_diag_initial_guess(N_states, hdiag):
    '''
    m is the amount of initial guesses
    '''
    hdiag = hdiag.reshape(-1,)
    V_size = hdiag.shape[0]
    Dsort = hdiag.argsort()
    energies = hdiag[Dsort][:N_states]*parameter.Hartree_to_eV
    V = np.zeros((V_size, N_states))
    for j in range(N_states):
        V[Dsort[j], j] = 1.0
    return V, energies

def TDA_diag_preconditioner(residual, sub_eigenvalue, hdiag ):
    '''
    DX - XΩ = r
    '''

    N_states = np.shape(residual)[1]
    t = 1e-14
    D = np.repeat(hdiag.reshape(-1,1), N_states, axis=1) - sub_eigenvalue
    '''
    force all small values not in [-t,t]
    '''
    D = np.where( abs(D) < t, np.sign(D)*t, D)
    new_guess = residual/D

    return new_guess

def TDDFT_diag_initial_guess(V_holder, W_holder, N_states, hdiag):
    '''
    return unchanged N_states just to keep consistent with other funcs
    '''
    V_holder[:,:N_states], energies = TDA_diag_initial_guess(
                                            N_states = N_states,
                                               hdiag = hdiag)

    return (V_holder, W_holder, N_states, energies,
                            V_holder[:,:N_states],
                            W_holder[:,:N_states])

def TDDFT_diag_preconditioner(R_x, R_y, omega, hdiag, conv_tol = None):
    '''
    preconditioners for each corresponding residual (state)
    '''
    hdiag = hdiag.reshape(-1,1)
    N_states = R_x.shape[1]
    t = 1e-14
    d = np.repeat(hdiag.reshape(-1,1), N_states, axis=1)

    D_x = d - omega
    D_x = np.where(abs(D_x) < t, np.sign(D_x)*t, D_x)
    D_x_inv = D_x**-1

    D_y = d + omega
    D_y = np.where(abs(D_y) < t, np.sign(D_y)*t, D_y)
    D_y_inv = D_y**-1

    X_new = R_x*D_x_inv
    Y_new = R_y*D_y_inv

    return X_new, Y_new

# def spolar_diag_initprec(RHS, hdiag=delta_hdiag2, conv_tol=None):
#
#     d = hdiag.reshape(-1,1)
#     RHS = RHS/d
#
#     return RHS
