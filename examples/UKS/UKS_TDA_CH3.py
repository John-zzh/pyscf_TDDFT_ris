# import numpy
from pyscf import gto,  dft,  tddft
from pyscf_TDDFT_ris import TDDFT_ris


mol         = gto.Mole()
mol.verbose = 3

mol.atom = '''
 C              0.24775912    0.61514033   -0.21841270
 H              0.75215095   -0.27653261    0.09046958
 H              0.75217056    1.32846722   -0.83617790
 H             -0.76104443    0.79348596    0.09046951
'''
mol.basis = 'def2-SVP'
mol.verbose = 3
mol.charge = 0
mol.spin = 1
mol.build()
print('mol.spin',mol.spin)
mf = dft.UKS(mol)

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
energies, X = td.kernel_TDA()
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
