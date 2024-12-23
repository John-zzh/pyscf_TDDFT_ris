
import numpy as np
from pyscf_TDDFT_ris import math_helper
from pyscf_TDDFT_ris import parameter
from scipy.sparse import csr_matrix
import time


def Davidson(matrix_vector_product,
                    hdiag,
                    N_states=20,
                    conv_tol=1e-5,
                    max_iter=25,
                    single=False ):
    '''
    AX = XΩ
    Davidson frame, can use different initial guess and preconditioner
    initial_guess is a function, takes the number of initial guess as input
    preconditioner is a function, takes the residual as the input
    '''
    print('====== Davidson Diagonalization Starts ======')

    D_start = time.time()

    A_size = hdiag.shape[0]
    print('size of A matrix =', A_size)
    size_old = 0
    size_new = min([N_states+8, 2*N_states, A_size])

    max_N_mv = max_iter*N_states + size_new
    V_holder = np.zeros((A_size, max_N_mv),dtype=np.float32 if single else np.float64)
    W_holder = np.zeros_like(V_holder)
    sub_A_holder = np.zeros((max_N_mv,max_N_mv),dtype=np.float32 if single else np.float64)
    '''
    generate the initial guesss and put into the basis holder V_holder
    '''
    V_holder = math_helper.TDA_diag_initial_guess(V_holder=V_holder, N_states=size_new, hdiag=hdiag)

    # V_holder[:,:size_new] = initial_vectors[:,:]

    subcost = 0
    MVcost = 0
    fill_holder_cost = 0
    subgencost = 0
    full_cost = 0
    print('step max||r||   sub_A.shape')
    for ii in range(max_iter):
    
        MV_start = time.time()
        W_holder[:, size_old:size_new] = matrix_vector_product(V_holder[:,size_old:size_new])
        MV_end = time.time()
        iMVcost = MV_end - MV_start
        MVcost += iMVcost

        subgencost_start = time.time()
        sub_A_holder = math_helper.gen_VW(sub_A_holder, V_holder, W_holder, size_old, size_new)
        sub_A = sub_A_holder[:size_new,:size_new]
        subgencost_end = time.time()
        subgencost += subgencost_end - subgencost_start
        sub_A = math_helper.utriangle_symmetrize(sub_A)
        # sub_A = math_helper.symmetrize(sub_A)

        '''
        Diagonalize the subspace Hamiltonian, and sorted.
        sub_eigenvalue[:N_states] are smallest N_states eigenvalues
        '''

        subcost_start = time.time()
        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        sub_eigenvalue = sub_eigenvalue[:N_states]
        sub_eigenket = sub_eigenket[:,:N_states]
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        full_cost_start = time.time()
        full_guess = np.dot(V_holder[:,:size_new], sub_eigenket)
        full_cost_end = time.time()
        full_cost += full_cost_end - full_cost_start

        AV = np.dot(W_holder[:,:size_new], sub_eigenket)
        residual = AV - full_guess * sub_eigenvalue

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)
        print('{:<3d}  {:<10.4e} {:<5d}'.format(ii+1, max_norm, sub_A.shape[0]))
        if max_norm < conv_tol or ii == (max_iter-1):
            break

        index = [r_norms.index(i) for i in r_norms if i>conv_tol]

        new_guess = math_helper.TDA_diag_preconditioner(residual = residual[:,index],
                                            sub_eigenvalue = sub_eigenvalue[index],
                                            hdiag = hdiag)

        fill_holder_cost_start = time.time()
        size_old = size_new
        V_holder, size_new = math_helper.Gram_Schmidt_fill_holder(V_holder, size_old, new_guess, double=True)
        fill_holder_cost_end = time.time()
        fill_holder_cost += fill_holder_cost_end - fill_holder_cost_start

    # energies = sub_eigenvalue*parameter.Hartree_to_eV

    D_end = time.time()
    Dcost = D_end - D_start

    if ii == max_iter-1:
        print('=== TDA Failed Due to Iteration Limit ===')
        print('current residual norms', r_norms)
        
    # print('energies:')
    # print(energies)
    print('Finished in {:d} steps, {:.2f} seconds'.format(ii+1, Dcost))
    print('Maximum residual norm = {:.2e}'.format(max_norm))
    print('Final subspace size = {:d}'.format(sub_A.shape[0]))
    for enrty in ['MVcost','fill_holder_cost','subgencost','subcost','full_cost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s {:<5.2%}".format(enrty, cost, cost/Dcost))
    print('========== Davidson Diagonalization Done ==========')
    return sub_eigenvalue, full_guess

def Davidson_Casida(matrix_vector_product,
                        hdiag,
                        N_states=20,
                        conv_tol=1e-5,
                        max_iter=25,
                        GS=False,
                        single=False ):
    '''
    [ A B ] X - [1   0] Y Ω = 0
    [ B A ] Y   [0  -1] X   = 0

    '''
    print('======= TDDFT Eigen Solver Statrs =======')

    davidson_start = time.time()
    A_size = hdiag.shape[0]
    print('size of A matrix =', A_size)
    size_old = 0
    size_new = min([N_states+8, 2*N_states, A_size])

    max_N_mv = (max_iter+1)*N_states
    
    '''
    [U1] = [A B][V]
    [U2]   [B A][W]

    U1 = AV + BW
    U2 = AW + BV

    a = [V.T W.T][A B][V] = [V.T W.T][U1] = VU1 + WU2
                 [B A][W]            [U2]
    '''
    V_holder = np.zeros((max_N_mv, A_size),dtype=np.float32 if single else np.float64)
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)

    VU1_holder = np.zeros((max_N_mv,max_N_mv),dtype=np.float32 if single else np.float64)
    VU2_holder = np.zeros_like(VU1_holder)
    WU1_holder = np.zeros_like(VU1_holder)
    WU2_holder = np.zeros_like(VU1_holder)

    VV_holder = np.zeros_like(VU1_holder)
    VW_holder = np.zeros_like(VU1_holder)
    WW_holder = np.zeros_like(VU1_holder)

    '''
    set up initial guess V= TDA initial guess, W=0
    '''

    V_holder = math_helper.TDA_diag_initial_guess(V_holder=V_holder,
                                                N_states=size_new,
                                                hdiag=hdiag)
    subcost = 0
    MVcost = 0
    fill_holder_cost = 0
    fill_holder_step_cost = 0
    subgencost = 0
    full_cost = 0
    # math_helper.show_memory_info('After Davidson initial guess set up')
    # print('step maximum residual norm')

    if GS == True:
        fill_holder = math_helper.VW_Gram_Schmidt_fill_holder
    else:
        fill_holder = math_helper.nKs_fill_holder

    for ii in range(max_iter):

        # print('size_old =', size_old)
        # print('size_new =', size_new)
        MV_start = time.time()

        U1_holder[size_old:size_new, :], U2_holder[size_old:size_new, :] = matrix_vector_product(
                                                                            X=V_holder[size_old:size_new, :],
                                                                            Y=W_holder[size_old:size_new, :])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        subgenstart = time.time()



        '''
        generate the subspace matrices
        sub_A = np.dot
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
        # pi = pi * -1
        omega, x, y = math_helper.TDDFT_subspace_eigen_solver(sub_A, sub_B, sigma, pi, N_states)
        # print('omega =', omega*parameter.Hartree_to_eV)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start
        # print(sub_A, sub_B, sigma, pi)
        '''
        compute the residual
        R_x = U1x + U2y - X_full*omega
        R_y = U2x + U1y + Y_full*omega
        X_full = Vx + Wy
        Y_full = Wx + Vy
        '''
        full_cost_start = time.time()
 
        V = V_holder[:size_new,:]
        W = W_holder[:size_new,:]
        U1 = U1_holder[:size_new, :]
        U2 = U2_holder[:size_new, :]

        X_full = x.T @ V + y.T @ W
        Y_full = x.T @ W + y.T @ V

        R_x = x.T @ U1 + y.T @ U2 - omega.reshape(-1, 1) * X_full
        R_y = x.T @ U2 + y.T @ U1 + omega.reshape(-1, 1) * Y_full

        full_cost_end = time.time()
        full_cost += full_cost_end - full_cost_start

        residual = np.hstack((R_x, R_y))
        # print('residual size =', residual.shape)
        r_norms = np.linalg.norm(residual, axis=1).tolist()
        # print('r_norms.shape', len(r_norms))
        max_norm = np.max(r_norms)
        # print('{:<3d}  {:<10.4e}'.format(ii+1, max_norm))
        print(f'iter: {ii+1:<3d}  max|R|: {max_norm:<10.2e} new_vectors: {size_new - size_old}  MVP: {MV_end - MV_start:.1f} seconds')

        if max_norm < conv_tol or ii == (max_iter -1):
            # math_helper.show_memory_info('After last Davidson iteration')
            break

        index = [r_norms.index(i) for i in r_norms if i > conv_tol]

        '''
        preconditioning step
        '''
        X_new, Y_new = math_helper.TDDFT_diag_preconditioner(R_x=R_x[index,:],
                                                            R_y=R_y[index,:],
                                                            omega=omega[index],
                                                            hdiag=hdiag)     
                                                   
        '''
        GS and symmetric orthonormalization
        '''
        size_old = size_new
        fill_holder_cost_start = time.time()
        V_holder, W_holder, size_new = fill_holder(V_holder=V_holder,
                                                    W_holder=W_holder,
                                                    X_new=X_new,
                                                    Y_new=Y_new,
                                                    m=size_old,
                                                    double=False)
        fill_holder_step_cost = time.time() - fill_holder_cost_start
        fill_holder_cost += fill_holder_step_cost

        if size_new == size_old:
            print('All new guesses kicked out!!!!!!!')
            break

    davidson_cost = time.time() - davidson_start

    if ii == (max_iter -1) and max_norm >= conv_tol:
        print('=== TDDFT eigen solver not converged due to max iteration mimit ===')
        print('max residual norms', np.max(r_norms))

    print(f'Finished in {ii+1:d} steps, {davidson_cost:.2f} seconds')
    print(f'final subspace = {sub_A.shape[0]}', )
    print(f'max_norm = {max_norm:.2e}')
    for enrty in ['MVcost','fill_holder_cost','subgencost','subcost','full_cost']:
        cost = locals()[enrty]
        print(f'{enrty:<10} {cost:<5.4f}s {cost/davidson_cost:<5.2%}')

    print('======= TDDFT Eigen Solver Done =======' )

    return omega, X_full, Y_full

