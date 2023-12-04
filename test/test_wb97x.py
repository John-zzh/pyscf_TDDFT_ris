import numpy as np
from pyscf import gto,  dft
from pyscf_TDDFT_ris import ris

def gen_mf(RKS, func, charge=0, spin=0):
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

    if RKS:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol) 

    # mf = mf.density_fit()
    mf.conv_tol = 1e-10
    mf.grids.level = 3
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

    nroots = 20
    for KS in ['RKS', 'UKS']:
    # for KS in ['RKS']:
        # for func in ['pbe', 'pbe0', 'wb97x']:
        for func in ['b3lyp']:
            for calc in ['TDA', 'TDDFT']:
            # for calc in ['TDDFT']:
                name = KS + '_' + func + '_' + calc
                print('======================================= {} {} {}-ris ======================================='.format(KS, func, calc))
                mf = gen_mf(RKS=True if KS=='RKS' else False, func=func, charge=0 if KS=='RKS' else 1, spin=0 if KS=='RKS' else 1)
                td = ris.TDDFT_ris(mf, nroots=nroots, spectra=False)
                if calc == 'TDA':
                    energies, X, oscillator_strength = td.kernel_TDA()
                elif  calc == 'TDDFT':
                    energies, X, Y, oscillator_strength = td.kernel_TDDFT()
                # print_list(energies)
                # print_list(oscillator_strength)
                print(name)
                # compare_nroors = min(nroots, 10)
                # diff(energies[:compare_nroors], res[name+"_ene"][:compare_nroors])
                # diff(oscillator_strength[:compare_nroors], res[name+"_spc"][:compare_nroors])
   
if __name__ == '__main__':
    main()
