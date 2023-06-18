from pyscf import dft
import numpy as np
from mokit.lib.gaussian import load_mol_from_fch, mo_fch2py
from mokit.lib.rwwfn import read_eigenvalues_from_fch, read_nbf_and_nif_from_fch
from pyscf_TDDFT_ris import TDDFT_ris

mol, mf = TDDFT_ris.get_mol_mf('2periacene_CAM-B3LYP.fch', 'CAM-B3LYP')

td = TDDFT_ris.TDDFT_ris(mf, mol, add_p=False, nroots = 20, max_iter=30)
energies, X, Y = td.kernel_TDDFT()
print('Excitations energies:')
print(energies)
print('==================')