from pyscf import dft
from pyscf_TDDFT_ris import TDDFT_ris
from pyscf.tools import molden
import numpy as np
np.set_printoptions(linewidth=250, threshold=np.inf)

def get_mol_mf(molden_file, functional):
    mol, mo_energy, mo_coeff, mo_occ, _, _ = molden.read(molden_file)

    mol.basis = 'def2-TZVP'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = functional
    mf.mo_occ = mo_occ
    mf.mo_coeff = mo_coeff
    mf.mo_energy = mo_energy

    return mol, mf

mol, mf = get_mol_mf(molden_file='molden.input', functional='pbe0')

td = TDDFT_ris.TDDFT_ris(mf, mol, add_p=False, nroots = 20)
energies, X, Y = td.kernel_TDDFT()
print(energies)
print('==================')

