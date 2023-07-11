from pyscf import dft
import sys
from mokit.lib.gaussian import load_mol_from_fch, mo_fch2py

fchname = 'O2.fch'
mol = load_mol_from_fch(fchname)
mf = dft.UKS(mol)
mf.xc = 'camb3lyp'
mf.grids.atom_grid = (99,590)
mf.verbose = 4
mf.max_cycle = 1
mf.kernel()

mf.mo_coeff = mo_fch2py(fchname)
dm = mf.make_rdm1()
mf.max_cycle = 10
mf.kernel(dm0=dm)
