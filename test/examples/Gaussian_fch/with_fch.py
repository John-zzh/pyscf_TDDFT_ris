
from pyscf_TDDFT_ris import TDDFT_ris, readMO

mf = readMO.get_mf_from_fch('2periacene_CAM-B3LYP.fch', 'CAM-B3LYP')


td = TDDFT_ris.TDDFT_ris(mf, add_p=False, nroots = 20, conv_tol=1e-3)
energies, X, Y = td.kernel_TDDFT()
