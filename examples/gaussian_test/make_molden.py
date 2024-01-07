import numpy as np
from pyscf import gto,  dft, tools
from pyscf_TDDFT_ris import TDDFT_ris, ris_pt2
import time

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
    # mf.conv_tol = 1e-10
    # mf.grids.level = 3
    mf.xc = func
    mol.charge = charge
    mol.spin = spin
    mf.kernel()
    return mf



mf = gen_mf(RKS=True, func='pbe0', charge=0, spin=0)
tools.molden.dump_scf(mf, 'methanol.molden')