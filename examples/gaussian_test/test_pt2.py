import numpy as np
from pyscf import gto,  dft
from pyscf_TDDFT_ris import TDDFT_ris, ris_pt2
import time

def gen_mf(RKS, func, charge=0, spin=0):
    mol = gto.Mole()
    mol.verbose = 3
    # mol.atom = '''
    # C         -4.89126        3.29770        0.00029
    # H         -5.28213        3.05494       -1.01161
    # O         -3.49307        3.28429       -0.00328
    # H         -5.28213        2.58374        0.75736
    # H         -5.23998        4.31540        0.27138
    # H         -3.22959        2.35981       -0.24953
    # '''

    mol.atom = '''
    C         -6.13841        3.45356        0.00000
    C         -4.81388        3.45356        0.00000
    H         -6.70072        2.59728        0.36431
    H         -6.70072        4.30985       -0.36431
    H         -4.25157        2.59728        0.36431
    H         -4.25157        4.30985       -0.36431
    '''

    mol.basis = 'def2-SVP'
    mol.build()

    if RKS:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol) 

    mf = mf.density_fit()
    # mf.conv_tol = 1e-10
    # mf.grids.level = 3
    mf.xc = func
    mol.charge = charge
    mol.spin = spin
    mf.kernel()
    return mf

def print_list(list):
    print('[', end='')
    for i in list:
        print(i, end=', ')
    print(']')
    print()

def diff(a,b):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = np.linalg.norm(a-b)
    print(diff, diff<1e-5)
    if diff > 1e-5:
        raise ValueError('wrong results generated')


def main():


    nroots = 5
    # for KS in ['RKS', 'UKS']:
    for KS in ['RKS']:
        # for func in ['pbe', 'pbe0', 'wb97x']:
        for func in ['pbe0']:
            # for calc in ['TDA', 'TDDFT']:
            # for calc in ['TDA']:
            for calc in ['TDDFT']:
                name = KS + '_' + func + '_' + calc
                print('======================================= {} {} {}-ris ======================================='.format(KS, func, calc))
                mf = gen_mf(RKS=True if KS=='RKS' else False, func=func, charge=0 if KS=='RKS' else 1, spin=0 if KS=='RKS' else 1)


                if calc == 'TDA':
                    start = time.time()
                    td = TDDFT_ris.TDDFT_ris(mf=mf, nroots=nroots, spectra=False)
                    energies, X, oscillator_strength = td.kernel_TDA()
                    end1 = time.time()
                    
                    print('#########################')

                    td = ris_pt2.TDDFT_ris_PT2(mf=mf, nroots=nroots, spectra=False,method='ris',spectra_window=10, single=True)
                    energy_CSF, X_SCF, oscillator_strength_SCF = td.kernel_TDA()
                    end2 = time.time()
                    print('davidson time:', end1-start)
                    print('PT2 time:', end2-end1)

                    print('davidson energy:', energies)
                    print('PT2      energy:', energy_CSF)
                if calc == 'TDDFT':

                    start = time.time()
                    td = TDDFT_ris.TDDFT_ris(mf=mf, nroots=nroots, spectra=False)
                    energies, X, Y, oscillator_strength = td.kernel_TDDFT()
                    end1 = time.time()
                    
                    print('#########################')

                    td = ris_pt2.TDDFT_ris_PT2(mf=mf, nroots=nroots, spectra=False,method='ris',spectra_window=10, single=True)
                    energy_CSF, X_SCF, Y_SCF, oscillator_strength_SCF = td.kernel_TDDFT()
                    end2 = time.time()
                    print('davidson time:', end1-start)
                    print('PT2 time:', end2-end1)

                    print('davidson energy:', energies)
                    print('PT2      energy:', energy_CSF)                 
              

   
if __name__ == '__main__':
    main()
