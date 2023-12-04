from pyscf import dft
from pyscf_TDDFT_ris import TDDFT_ris, readMO

mf = readMO.get_mf_from_molden(molden_file='Fe_turbo_molden.input', functional='pbe0', basis='def2-tzvp')

# td = TDDFT_ris.TDDFT_ris(mf, mol, add_p=False, nroots = 20)
# energies, X, Y = td.kernel_TDDFT()
# print(energies)
# print('==================')

td = TDDFT_ris.TDDFT_ris_pt2(mf, nroots=20, spectra=False)
td.kernel()

