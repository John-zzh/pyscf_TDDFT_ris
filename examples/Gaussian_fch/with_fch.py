
from pyscf_TDDFT_ris import TDDFT_ris

mol, mf = TDDFT_ris.get_mol_mf_fch('2periacene_CAM-B3LYP.fch', 'CAM-B3LYP')

td = TDDFT_ris.TDDFT_ris(mf, mol, add_p=False, nroots = 20, max_iter=30)
energies, X, Y = td.kernel_TDDFT()
print('Excitations energies:')
print(energies)
print('==================')