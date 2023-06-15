# import numpy
from pyscf import gto,  dft,  tddft
from PySCF-TDDFT-ris.TDDFT_ris import TDDFT_ris


mol         = gto.Mole()
mol.verbose = 3
mol.atom    = '''
H       -0.9450370725    -0.0000000000     1.1283908757
C       -0.0000000000     0.0000000000     0.5267587663
H        0.9450370725     0.0000000000     1.1283908757
O        0.0000000000    -0.0000000000    -0.6771667936
'''
# mol.atom = '''
# C         -4.89126        3.29770        0.00029
# H         -5.28213        3.05494       -1.01161
# O         -3.49307        3.28429       -0.00328
# H         -5.28213        2.58374        0.75736
# H         -5.23998        4.31540        0.27138
# H         -3.22959        2.35981       -0.24953
# '''
mol.basis = 'def2-SVP'
mol.build()

mf = dft.RKS(mol)
mf = mf.density_fit()
mf.xc = "pbe0"
mf.conv_tol = 1e-10
mf.grids.level = 3
mf.kernel()

# td = tddft.TDA(mf)
# td.nroots = 5
# td.kernel()
# print('==================')

# td = tddft.TDDFT(mf)
# td.nroots = 5
# td.kernel()
# print('==================')
# print('==================')
'''
to invoke the TDDFT-ris method
'''

td = TDDFT_ris.TDDFT_ris(mf, mol, add_p=False, theta=0.2, nroots = 20)
energies, X, Y = td.kernel_TDDFT()
print(energies)
print('==================')


'''
compar with standard (ab-initio) TDDFT energies
pyscf built-in davidson is problematic, so here is just  a hack:
I export the ab-initio TDDFT matrix-vector product function, and use my own
eigen_solver to compute the energy
'''
TD = tddft.TDDFT(mf)
TDDFT_vind, Hdiag = TD.gen_vind(mf)
TD = TDDFT_ris.TDDFT_ris(mf, mol, pyscf_TDDFT_vind=TDDFT_vind, nroots = 20)
energies, X, Y = TD.kernel_TDDFT()
print(energies)
print('==================')
