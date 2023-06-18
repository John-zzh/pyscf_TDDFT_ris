from pyscf import dft
import numpy as np
from mokit.lib.gaussian import load_mol_from_fch, mo_fch2py
from mokit.lib.rwwfn import read_eigenvalues_from_fch, read_nbf_and_nif_from_fch
from pyscf_TDDFT_ris import TDDFT_ris


def get_mol_mf(fch_file, functional):

    mol = load_mol_from_fch(fch_file)
    mol.build()
    print('======= Molecular Coordinates ========')
    print(mol.atom)

    [print(k, v) for k, v in mol.basis.items()]
    mf = dft.RKS(mol)
    mf.mo_coeff = mo_fch2py(fch_file)
    nbf, nif = read_nbf_and_nif_from_fch(fch_file)
    mf.mo_energy = read_eigenvalues_from_fch(fch_file, nif=nif, ab='alpha')
    nocc = mol.nelectron // 2
    mf.mo_occ = np.asarray([2] * nocc + [0] * (nbf - nocc))
    mf.xc = functional
    return mol, mf

mol, mf = get_mol_mf('2periacene_CAM-B3LYP.fch', 'CAM-B3LYP')
td = TDDFT_ris.TDDFT_ris(mf, mol, add_p=False, nroots = 20, max_iter=30)
energies, X, Y = td.kernel_TDDFT()
print('Excitations energies:')
print(energies)
print('==================')