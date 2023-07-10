#!/usr/bin/python

import numpy as np
import scipy
import time

import multiprocessing as mp

num_cores = int(mp.cpu_count())
print("This job can use: " + str(num_cores) + "CPUs")

def commutator(A,B):
    commu = np.dot(A,B) - np.dot(B,A)
    return commu

def cond_number(A):
    s,u = np.linalg.eig(A)
    s = abs(s)
    cond = max(s)/min(s)
    return cond

def matrix_power(S,a):
    '''X == S^a'''
    s,ket = np.linalg.eigh(S)
    s = s**a
    X = np.dot(ket*s,ket.T)
    return X

def copy_array(A):
    B = np.zeros_like(A)
    dim = len(B.shape)
    if dim == 1:
        B[:,] = A[:,]
    elif dim == 2:
        B[:,:] = A[:,:]
    elif dim == 3:
        B[:,:,:] = A[:,:,:]
    return B

def Gram_Schmidt_bvec(A, bvec):
    '''orthonormalize vector b against all vectors in A
       b = b - A*(A.T*b)
       suppose A is orthonormalized
    '''
    if A.shape[1] != 0:
        projections_coeff = np.dot(A.T, bvec)
        bvec -= np.dot(A, projections_coeff)
    return bvec

def VW_Gram_Schmidt(x, y, V, W):
    '''orthonormalize vector |x,y> against all vectors in |V,W>'''
    m = np.dot(V.T,x) + np.dot(W.T,y)

    n = np.dot(W.T,x) + np.dot(V.T,y)

    x = x - np.dot(V,m) - np.dot(W,n)

    y = y - np.dot(W,m) - np.dot(V,n)
    return x, y

def block_symmetrize(A,m,n):
    A[m:n,:m] = A[:m,m:n].T
    return A

def gen_anisotropy(a):

    # a = 0.5*(a.T + a)
    # tr = (1.0/3.0)*np.trace(a)
    # xx = a[0,0]
    # yy = a[1,1]
    # zz = a[2,2]

    # xy = a[0,1]
    # xz = a[0,2]
    # yz = a[1,2]
    # anis = (xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2 + 6*(xz**2 + xy**2 + yz**2)
    # anis = 0.5*anis
    # anis = anis**0.5
    a = 0.5*(a.T + a)
    tr = (1.0/3.0)*np.trace(a)
    # print('type(tr)', type(tr))
    xx = a[0,0]
    # print('type(xx)', type(xx))
    yy = a[1,1]
    zz = a[2,2]

    xy = a[0,1]
    xz = a[0,2]
    yz = a[1,2]

    ssum = xx**2 + yy**2 + zz**2 + 2*(xy**2 + xz**2 + yz**2)
    anis = (1.5 * abs(ssum - 3*tr**2))**0.5
    return float(tr), float(anis)

def utriangle_symmetrize(A):
    upper = np.triu_indices(n=A.shape[0], k=1)
    lower = (upper[1], upper[0])
    A[lower] = A[upper]
    return A

def anti_block_symmetrize(A,m,n):
    A[m:n,:m] = -A[:m,m:n].T
    return A

def gen_VW(sub_A_holder, V_holder, W_holder, size_old, size_new, symmetry = True, up_triangle = False):
    '''
    [ V_old.T ] [W_old, W_new] = [VW_old,        V_old.T W_new] = [VW_old, V_current.T W_new]
    [ V_new.T ]                  [V_new.T W_old, V_new.T W_new]   [               '' ''     ]


    V_holder or W_holder

                size_old     size_new
    |--------------|--------------|------------|
    |              |              |            |
    |   V_old      |    V_new     |            |
    |              |              |            |
    |              |              |            |
    |              |              |            |
    |              |              |            |
    |        [ V_current ]        |            |
    |              |              |            |
    |              |              |            |
    |              |              |            |
    |              |              |            |
    |              |              |            |
    |              |              |            |
    |              |              |            |
    |              |              |            |
    |--------------|--------------|------------|

    sub_A_holder

                            size_old            size_new
                |---------------|-----------------｜-----------------｜
                |               |                 ｜                 ｜
                |    VW_old     |                 ｜                 ｜
                |               |                 ｜                 ｜
      size_old  |---------------|V_current.T W_new｜-----------------｜
                | V_new.T W_old |                 ｜                 ｜
                | or            |                 ｜                 ｜
                | W_new.T V_old |                 ｜                 ｜
      size_new  |---------------|-----------------｜-----------------｜
                |               |                 ｜                 ｜
                |               |                 ｜                 ｜
                |               |                 ｜                 ｜
                |---------------|-----------------｜-----------------｜
    '''

    V_current = V_holder[:,:size_new]
    W_new = W_holder[:,size_old:size_new]
    sub_A_holder[:size_new,size_old:size_new] = np.dot(V_current.T, W_new)

    if symmetry == True:
        sub_A_holder = block_symmetrize(sub_A_holder,size_old,size_new)
    elif symmetry == False:
        if up_triangle == False:
            '''
            up_triangle == False means also explicitly compute the lower triangle,
                                        either equal upper triangle.T or recompute
            '''
            V_new = V_holder[:,size_old:size_new]
            W_old = W_holder[:,:size_old]
            sub_A_holder[size_old:size_new,:size_old] = np.dot(V_new.T, W_old)
        elif up_triangle == True:
            '''
            otherwise juts let the lower triangle be zeros
            '''
            pass

    return sub_A_holder


def gen_VP(sub_P_holder, V_holder, P, size_old, size_new):
    '''
    [ V_old.T ] [P] = [P_old    ]
    [ V_new.T ]       [V_new.T P]
    '''
    V_new = V_holder[:,size_old:size_new]
    sub_P_holder[size_old:size_new,:] = np.dot(V_new.T, P)
    return sub_P_holder


def gen_sub_pq(V_holder, W_holder, P, Q, VP_holder, WQ_holder, WP_holder, VQ_holder, size_old, size_new):
    '''
    [ V_old.T ] [P_old] = [P_old]
    [ V_new.T ]           [V_new.T [P_old]]
    '''
    VP_holder = gen_VP(VP_holder, V_holder, P, size_old, size_new)

    VQ_holder = gen_VP(VQ_holder, V_holder, Q, size_old, size_new)

    WP_holder = gen_VP(WP_holder, W_holder, P, size_old, size_new)

    WQ_holder = gen_VP(WQ_holder, W_holder, Q, size_old, size_new)

    p = VP_holder[:size_new,:] + WQ_holder[:size_new,:]
    q = WP_holder[:size_new,:] + VQ_holder[:size_new,:]

    return p, q, VP_holder, WQ_holder, WP_holder, VQ_holder


def gen_sub_ab(V_holder, W_holder, U1_holder, U2_holder,
              VU1_holder, WU2_holder, VU2_holder, WU1_holder,
              VV_holder, WW_holder, VW_holder,
              size_old, size_new):
    '''
    a = V.T U1 + W.T U2
    b = V.T U2 + W.T U1
    V.T U1 = gen_sub_A()
    '''

    VU1_holder = gen_VW(VU1_holder, V_holder, U1_holder, size_old, size_new, symmetry = False, up_triangle = True)
    VU2_holder = gen_VW(VU2_holder, V_holder, U2_holder, size_old, size_new, symmetry = False, up_triangle = True)
    WU1_holder = gen_VW(WU1_holder, W_holder, U1_holder, size_old, size_new, symmetry = False, up_triangle = True)
    WU2_holder = gen_VW(WU2_holder, W_holder, U2_holder, size_old, size_new, symmetry = False, up_triangle = True)

    VV_holder = gen_VW(VV_holder, V_holder, V_holder, size_old, size_new, symmetry = False, up_triangle = True)
    WW_holder = gen_VW(WW_holder, W_holder, W_holder, size_old, size_new, symmetry = False, up_triangle = True)
    VW_holder = gen_VW(VW_holder, V_holder, W_holder, size_old, size_new, symmetry = False, up_triangle = False)

    sub_A = VU1_holder[:size_new, :size_new] + WU2_holder[:size_new, :size_new]
    sub_A = utriangle_symmetrize(sub_A)

    sub_B = VU2_holder[:size_new, :size_new] + WU1_holder[:size_new, :size_new]
    sub_B = utriangle_symmetrize(sub_B)

    sigma = VV_holder[:size_new, :size_new] - WW_holder[:size_new, :size_new]
    sigma = utriangle_symmetrize(sigma)

    pi = VW_holder[:size_new, :size_new] - VW_holder[:size_new, :size_new].T

    return sub_A, sub_B, sigma, pi, VU1_holder, WU2_holder, VU2_holder, WU1_holder, VV_holder, WW_holder, VW_holder






def Gram_Schmidt_fill_holder(V, count, vecs, double = False):
    '''V is a vectors holder
       count is the amount of vectors that already sit in the holder
       nvec is amount of new vectors intended to fill in the V
       count will be final amount of vectors in V
    '''
    nvec = np.shape(vecs)[1]
    for j in range(nvec):
        vec = vecs[:, j].reshape(-1,1)
        vec = Gram_Schmidt_bvec(V[:, :count], vec)   #single orthonormalize
        if double == True:
            vec = Gram_Schmidt_bvec(V[:, :count], vec)   #double orthonormalize
        norm = np.linalg.norm(vec)
        if  norm > 1e-14:
            vec = vec/norm
            V[:, count] = vec[:,0]
            count += 1
    new_count = count
    return V, new_count

def S_symmetry_orthogonal(x,y):
    '''symmetrically orthogonalize the vectors |x,y> and |y,x>
       as close to original vectors as possible
    '''
    x_p_y = x + y
    x_p_y_norm = np.linalg.norm(x_p_y)

    x_m_y = x - y
    x_m_y_norm = np.linalg.norm(x_m_y)

    a = x_p_y_norm/x_m_y_norm

    x_p_y /= 2
    x_m_y *= a/2

    new_x = x_p_y + x_m_y
    new_y = x_p_y - x_m_y

    return new_x, new_y

def symmetrize(A):
    A = (A + A.T)/2
    return A

def anti_symmetrize(A):
    A = (A - A.T)/2
    return A

def check_orthonormal(A):
    '''
    define the orthonormality of a matrix A as the norm of (A.T*A - I)
    '''
    n = np.shape(A)[1]
    B = np.dot(A.T, A)
    c = np.linalg.norm(B - np.eye(n))
    return c

def check_symmetry(A):
    '''
    define matrix A is symmetric
    '''
    a = np.linalg.norm(A - A.T)
    return a

def check_anti_symmetry(A):
    '''
    define matrix A is symmetric
    '''
    a = np.linalg.norm(A + A.T)
    return a

def VW_Gram_Schmidt_fill_holder(V_holder, W_holder, m, X_new, Y_new, double = False):
    '''
    put X_new into V, and Y_new into W
    m: the amount of vectors that already on V or W
    nvec: amount of new vectors intended to put in the V and W
    '''
    VWGSstart = time.time()
    nvec = np.shape(X_new)[1]

    GScost = 0
    normcost = 0
    symmetrycost = 0
    GSfill_start = time.time()
    for j in range(0, nvec):
        V = V_holder[:,:m]
        W = W_holder[:,:m]

        x_tmp = X_new[:,j].reshape(-1,1)
        y_tmp = Y_new[:,j].reshape(-1,1)

        VW_Gram_Schmidt_start = time.time()
        x_tmp,y_tmp = VW_Gram_Schmidt(x_tmp, y_tmp, V, W)
        if double == True:
            # print('double')
            x_tmp,y_tmp = VW_Gram_Schmidt(x_tmp, y_tmp, V, W)
        VW_Gram_Schmidt_end = time.time()
        GScost += VW_Gram_Schmidt_end - VW_Gram_Schmidt_start

        symmetry_start = time.time()
        x_tmp,y_tmp = S_symmetry_orthogonal(x_tmp,y_tmp)
        symmetry_end = time.time()
        symmetrycost += symmetry_end - symmetry_start

        norm_start = time.time()
        xy_norm = (np.dot(x_tmp.T, x_tmp)+np.dot(y_tmp.T, y_tmp))**0.5


        if  xy_norm > 1e-14:
            x_tmp = x_tmp/xy_norm
            y_tmp = y_tmp/xy_norm

            V_holder[:,m] = x_tmp[:,0]
            W_holder[:,m] = y_tmp[:,0]
            m += 1
        else:
            print('vector kicked out during GS orthonormalization')

        norm_end = time.time()
        normcost += norm_end - norm_start

    GSfill_end = time.time()
    GSfill_cost = GSfill_end - GSfill_start

    # print('GScost = {:.2%}'.format(GScost/GSfill_cost))
    # print('symmetrycost ={:.2%}'.format(symmetrycost/GSfill_cost))
    # print('normcost ={:.2%}'.format(normcost/GSfill_cost))
    new_m = m

    return V_holder, W_holder, new_m

def solve_AX_Xla_B(A, omega, Q):
    '''AX - XΩ  = Q
       A, Ω, Q are known, solve X
    '''
    Qnorm = np.linalg.norm(Q, axis=0, keepdims = True)
    Q = Q/Qnorm
    N_vectors = len(omega)
    a, u = np.linalg.eigh(A)
    # print('a =',a)
    # print('omega =',omega)
    ub = np.dot(u.T, Q)
    ux = np.zeros_like(Q)
    for k in range(N_vectors):
        ux[:, k] = ub[:, k]/(a - omega[k])
    X = np.dot(u, ux)
    X *= Qnorm

    return X

def TDDFT_subspace_eigen_solver(a, b, sigma, pi, k):
    ''' [ a b ] x - [ σ   π] x  Ω = 0 '''
    ''' [ b a ] y   [-π  -σ] y    = 0 '''

    d = abs(np.diag(sigma))
    d_mh = d**(-0.5)

    s_m_p = d_mh.reshape(-1,1) * (sigma - pi) * d_mh.reshape(1,-1)

    '''LU = d^−1/2 (σ − π) d^−1/2'''
    ''' A = PLU '''
    ''' if A is diagonally dominant, P is identity matrix (in fact not always) '''
    P_permutation, L, U = scipy.linalg.lu(s_m_p)

    L = np.dot(P_permutation, L)

    L_inv = np.linalg.inv(L)
    U_inv = np.linalg.inv(U)

    # L_inv = np.linalg.cholesky(np.linalg.inv(s_m_p))
    # U_inv = L_inv.T
    ''' a ̃−b ̃= U^-T d^−1/2 (a−b) d^-1/2 U^-1 = GG^T '''
    dambd =  d_mh.reshape(-1,1)*(a-b)*d_mh.reshape(1,-1)
    GGT = np.linalg.multi_dot([U_inv.T, dambd, U_inv])

    G = scipy.linalg.cholesky(GGT, lower=True)
    G_inv = np.linalg.inv(G)

    ''' M = G^T L^−1 d^−1/2 (a+b) d^−1/2 L^−T G '''
    dapbd = d_mh.reshape(-1,1)*(a+b)*d_mh.reshape(1,-1)
    M = np.linalg.multi_dot([G.T, L_inv, dapbd, L_inv.T, G])

    omega2, Z = np.linalg.eigh(M)
    omega = (omega2**0.5)[:k]
    Z = Z[:,:k]

    ''' It requires Z^T Z = 1/Ω '''
    ''' x+y = d^−1/2 L^−T GZ Ω^-0.5 '''
    ''' x−y = d^−1/2 U^−1 G^−T Z Ω^0.5 '''

    x_p_y = d_mh.reshape(-1,1)\
            *np.linalg.multi_dot([L_inv.T, G, Z])\
            *(np.array(omega)**-0.5).reshape(1,-1)

    x_m_y = d_mh.reshape(-1,1)\
            *np.linalg.multi_dot([U_inv, G_inv.T, Z])\
            *(np.array(omega)**0.5).reshape(1,-1)

    x = (x_p_y + x_m_y)/2
    y = x_p_y - x

    return omega, x, y

def TDDFT_subspace_liear_solver(a, b, sigma, pi, p, q, omega):
    '''[ a b ] x - [ σ   π] x  Ω = p
       [ b a ] y   [-π  -σ] y    = q
       normalize the right hand side first
    '''
    pq = np.vstack((p,q))
    pqnorm = np.linalg.norm(pq, axis=0, keepdims = True)

    p = p/pqnorm
    q = q/pqnorm

    d = abs(np.diag(sigma))
    d_mh = d**(-0.5)

    '''LU = d^−1/2 (σ − π) d^−1/2
       A = PLU
       P is identity matrix only when A is diagonally dominant
    '''
    s_m_p = d_mh.reshape(-1,1) * (sigma - pi) * d_mh.reshape(1,-1)
    P_permutation, L, U = scipy.linalg.lu(s_m_p)
    L = np.dot(P_permutation, L)

    L_inv = np.linalg.inv(L)
    U_inv = np.linalg.inv(U)

    p_p_q_tilde = np.dot(L_inv, d_mh.reshape(-1,1)*(p+q))
    p_m_q_tilde = np.dot(U_inv.T, d_mh.reshape(-1,1)*(p-q))

    ''' a ̃−b ̃= U^-T d^−1/2 (a−b) d^-1/2 U^-1 = GG^T'''
    dambd = d_mh.reshape(-1,1)*(a-b)*d_mh.reshape(1,-1)
    GGT = np.linalg.multi_dot([U_inv.T, dambd, U_inv])

    '''G is lower triangle matrix'''
    G = scipy.linalg.cholesky(GGT, lower=True)
    G_inv = np.linalg.inv(G)

    '''a ̃+ b ̃= L^−1 d^−1/2 (a+b) d^−1/2 L^−T
       M = G^T (a ̃+ b ̃) G
    '''
    dapba = d_mh.reshape(-1,1)*(a+b)*d_mh.reshape(1,-1)
    a_p_b_tilde = np.linalg.multi_dot([L_inv, dapba, L_inv.T])
    M = np.linalg.multi_dot([G.T, a_p_b_tilde, G])
    T = np.dot(G.T, p_p_q_tilde)
    T += np.dot(G_inv, p_m_q_tilde * omega.reshape(1,-1))

    Z = solve_AX_Xla_B(M, omega**2, T)

    '''(x ̃+ y ̃) = GZ
       x + y = d^-1/2 L^-T (x ̃+ y ̃)
       x - y = d^-1/2 U^-1 (x ̃- y ̃)
    '''
    x_p_y_tilde = np.dot(G,Z)
    x_p_y = d_mh.reshape(-1,1) * np.dot(L_inv.T, x_p_y_tilde)

    x_m_y_tilde = (np.dot(a_p_b_tilde, x_p_y_tilde) - p_p_q_tilde)/omega
    x_m_y = d_mh.reshape(-1,1) * np.dot(U_inv, x_m_y_tilde)

    x = (x_p_y + x_m_y)/2
    y = x_p_y - x
    x *= pqnorm
    y *= pqnorm
    return x, y
