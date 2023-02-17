import numpy
from pyscf import gto, dft, tdscf, tddft
from TDDFT_ris import TDDFT_ris


mol         = gto.Mole()
mol.verbose = 3
mol.atom    = '''
H       -0.9450370725    -0.0000000000     1.1283908757
C       -0.0000000000     0.0000000000     0.5267587663
H        0.9450370725     0.0000000000     1.1283908757
O        0.0000000000    -0.0000000000    -0.6771667936
'''
mol.basis = 'def2-SVP'
mol.build()

mf    = dft.RKS(mol)
mf.xc = "pbe0"
mf.conv_tol = 1e-10
mf.kernel()

td = tddft.TDA(mf)
td.nroots = 5
td.kernel()
print('==================')

td = tddft.TDDFT(mf)
td.nroots = 5
td.kernel()
print('==================')
print('==================')

td = TDDFT_ris.TDDFT_ris(mf, mol, add_p=False, theta=0.2)
td.nroots = 5
converged, e, amps = td.kernel_TDA()
print(e)

# td = TDDFT_ris.TDDFT_ris(mf, mol, add_p=False, theta=0.2)
print('==================')
converged, e, amps = td.kernel_TDDFT()
print(e)
