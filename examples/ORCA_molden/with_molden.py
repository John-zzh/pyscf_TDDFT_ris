from pyscf import dft
from pyscf_TDDFT_ris import TDDFT_ris

mol, mf = TDDFT_ris.get_mol_mf_molden(molden_file='methanol.molden.input', functional='pbe0', basis='def2-svp')

td = TDDFT_ris.TDDFT_ris(mf, mol, add_p=False, nroots = 20)
energies, X, Y = td.kernel_TDDFT()
print(energies)
print('==================')

