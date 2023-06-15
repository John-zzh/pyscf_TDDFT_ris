from pyscf import gto, dft, tddft
from pyscf_TDDFT_ris import TDDFT_ris
from pyscf.tools import molden
import numpy as np
np.set_printoptions(linewidth=500, precision=2)
# mol         = gto.Mole()
# mol.verbose = 3
# mol.atom    = '''
# H       -0.9450370725    -0.0000000000     1.1283908757
# C       -0.0000000000     0.0000000000     0.5267587663
# H        0.9450370725     0.0000000000     1.1283908757
# O        0.0000000000    -0.0000000000    -0.6771667936
# '''

# mol.basis = 'def2-SVP'
# mol.build()



mol, mo_energy, mo_coeff, mo_occ, _, _ = molden.read('methanol.molden.input')
# mol, mo_energy, mo_coeff, mo_occ, _, _ = molden.read('molden.input')
# print(mol.atom)
# print(mol.basis)
print('mo_energy')
print(mo_energy.shape)

# print(mo_occ)

mf = dft.RKS(mol)
mf.xc = "pbe0"
mf.mo_occ = mo_occ
mf.mo_coeff = mo_coeff
mf.mo_energy = mo_energy
print('mf.mo_coeff.shape')
print(mf.mo_coeff.shape)


td = TDDFT_ris.TDDFT_ris(mf, mol, add_p=False, theta=0.2, nroots = 20)
energies, X, Y = td.kernel_TDDFT()
print(energies)
print('==================')

# TD = TDDFT_ris.TDDFT_ris(mf, mol, pyscf_TDDFT_vind=TDDFT_vind, nroots = 20)
# energies, X, Y = TD.kernel_TDDFT()

# print(energies)
# print('==================')