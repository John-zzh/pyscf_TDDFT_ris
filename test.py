import numpy as np
from pyscf import gto,  dft
from pyscf_TDDFT_ris import TDDFT_ris

def gen_mol():
    mol = gto.Mole()
    mol.verbose = 3
    mol.atom = '''
    C         -4.89126        3.29770        0.00029
    H         -5.28213        3.05494       -1.01161
    O         -3.49307        3.28429       -0.00328
    H         -5.28213        2.58374        0.75736
    H         -5.23998        4.31540        0.27138
    H         -3.22959        2.35981       -0.24953
    '''
    mol.basis = 'def2-SVP'
    mol.build()
    return mol

def gen_mf(mol, RKS, func, charge=0, spin=0):
    if RKS:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)

    # mf = mf.density_fit()
    # mf.conv_tol = 1e-10
    # mf.grids.level = 3
    mf.xc = func
    mol.charge = charge
    mol.spin = spin
    mf.kernel()
    return mf

def main():
    mol = gen_mol()
    '''
        RKS 
    '''


    ''' pbe '''
    print('======================================= RKS pbe TDA-ris =======================================')
    mf = gen_mf(mol=mol, RKS=True, func='tpss', charge=0, spin=0)
    td = TDDFT_ris.TDDFT_ris(mf, nroots = 10)
    energies, X, oscillator_strength = td.kernel_TDA()
    standard = []

    print('======================================= RKS pbe TDDFT-ris =======================================')
    td = TDDFT_ris.TDDFT_ris(mf, nroots = 10)
    energies, X, Y, oscillator_strength = td.kernel_TDDFT()
    standard = []  


    ''' pbe0 '''
    print('======================================= RKS pbe0 TDA-ris =======================================')
    mf = gen_mf(mol=mol, RKS=True, func='pbe0', charge=0, spin=0)
    td = TDDFT_ris.TDDFT_ris(mf, nroots = 10)
    energies, X, oscillator_strength = td.kernel_TDA()
    standard = []

    print('======================================= RKS pbe0 TDDFT-ris =======================================')
    energies, X, Y, oscillator_strength = td.kernel_TDDFT()
    standard = []


    ''' wb97x '''
    print('======================================= RKS wb97x TDA-ris =======================================')
    mf = gen_mf(mol=mol, RKS=True, func='wb97x', charge=0, spin=0)
    td = TDDFT_ris.TDDFT_ris(mf, nroots = 10)
    energies, X, oscillator_strength = td.kernel_TDA()
    standard = []

    print('======================================= RKS wb97x TDDFT-ris =======================================')
    td = TDDFT_ris.TDDFT_ris(mf, nroots = 10)
    energies, X, Y, oscillator_strength = td.kernel_TDDFT()
    standard = []   




    '''
        UKS 
    '''
    charge = 1
    ''' pbe '''
    print('======================================= UKS pbe TDA-ris =======================================')
    mf = gen_mf(mol=mol, RKS=False, func='tpss', charge=charge, spin=1)
    td = TDDFT_ris.TDDFT_ris(mf, nroots = 10)
    energies, X, oscillator_strength = td.kernel_TDA()
    standard = []

    print('======================================= UKS pbe TDDFT-ris =======================================')
    td = TDDFT_ris.TDDFT_ris(mf, nroots = 10)
    energies, X, Y, oscillator_strength = td.kernel_TDDFT()
    standard = []  

    ''' pbe0 '''
    print('======================================= UKS pbe0 TDA-ris =======================================')
    mf = gen_mf(mol=mol, RKS=False, func='pbe0', charge=charge, spin=1)
    td = TDDFT_ris.TDDFT_ris(mf, nroots = 10)
    energies, X, oscillator_strength = td.kernel_TDA()
    standard = []

    print('======================================= UKS pbe0 TDDFT-ris =======================================')
    energies, X, Y, oscillator_strength = td.kernel_TDDFT()
    standard = []


    ''' wb97x '''
    print('====================================== UKS wb97x TDA-ris =======================================')
    mf = gen_mf(mol=mol, RKS=False, func='wb97x', charge=charge, spin=1)
    td = TDDFT_ris.TDDFT_ris(mf, nroots = 10)
    energies, X, oscillator_strength = td.kernel_TDA()
    standard = []

    print('======================================= UKS wb97x TDDFT-ris =======================================')
    td = TDDFT_ris.TDDFT_ris(mf, nroots = 10)
    energies, X, Y, oscillator_strength = td.kernel_TDDFT()
    standard = []   

    
if __name__ == '__main__':
    main()
