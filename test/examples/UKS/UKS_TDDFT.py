# import numpy
from pyscf import gto,  dft,  tddft
from pyscf_TDDFT_ris import TDDFT_ris


mol         = gto.Mole()
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
mol.verbose = 3
mol.build()

mf = dft.UKS(mol)
mol.charge = 1
mol.spin = 1
mf = mf.density_fit()
mf.xc = "pbe0"
mf.conv_tol = 1e-9
mf.grids.level = 3
mf.kernel()

# print('mf.mol == mol', mf.mol == mol)
print('mf.mo_coeff.shape =', mf.mo_coeff.shape)
print('mf.mo_energy.shape =',mf.mo_energy.shape)
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

# ab = 0

# mf.mo_occ = mf.mo_occ[ab,:]
# mf.mo_coeff = mf.mo_coeff[ab,:]
# mf.mo_energy = mf.mo_energy[ab,:]

# print('mf.mo_occ',mf.mo_occ)

td = TDDFT_ris.TDDFT_ris(mf, add_p=False, theta=0.2, nroots = 10)
energies, X, Y, oscillator_strength = td.kernel_TDDFT()
# print(energies)
print('==================')


'''
compar with standard (ab-initio) TDDFT energies
pyscf built-in davidson is problematic, so here is just  a hack:
I export the ab-initio TDDFT matrix-vector product function, and use my own
eigen_solver to compute the energy
'''
# TD = tddft.TDDFT(mf)
# TDDFT_vind, Hdiag = TD.gen_vind(mf)
# TD = TDDFT_ris.TDDFT_ris(mf, mol, pyscf_TDDFT_vind=TDDFT_vind, nroots = 20)
# energies, X, Y = TD.kernel_TDDFT()
# print(energies)
# print('==================')
