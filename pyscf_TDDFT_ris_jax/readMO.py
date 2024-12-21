from pyscf import scf, dft
import numpy as np
import os

def get_mf_from_fch(fch_file: str, functional: str = None):
    '''
    fch_file: fch file generated by Gaussian
    functional: three options
        1) functional name used in Gaussian 
        2) a_x value, the HF componen in hybrid functional
        3) (omega, RSH_alpha, RSH_beta), the range-separated parameter in RSH functional
    '''
    from mokit.lib.gaussian import load_mol_from_fch, mo_fch2py
    from mokit.lib.rwwfn import read_eigenvalues_from_fch, read_nbf_and_nif_from_fch, read_na_and_nb_from_fch
    
    mol = load_mol_from_fch(fch_file, path=os.getcwd())
    mo_coeff = mo_fch2py(fch_file)
    nbf, nif = read_nbf_and_nif_from_fch(fch_file)
    print('nbf =', nbf)
    print('nif =', nif)

    mol.build(parse_arg = False)
    # print('======= Molecular Coordinates ========')
    # print(mol.atom)
    # print('======= Basis Set ========')
    # [print(k, v) for k, v in mol.basis.items()]

    if isinstance(mo_coeff, tuple):
        mo_coeff = np.asarray(mo_coeff)
        mo_energy_a = read_eigenvalues_from_fch(fch_file, nif=nif, ab='alpha')
        mo_energy_b = read_eigenvalues_from_fch(fch_file, nif=nif, ab='beta')
        mo_energy =  np.asarray((mo_energy_a, mo_energy_b))
    else:
        mo_energy = read_eigenvalues_from_fch(fch_file, nif=nif, ab='alpha')

    print('mo_coeff.shape', mo_coeff.shape)
    print('mo_energy.shape', mo_energy.shape)

    if mo_coeff.ndim == 2:
        print('Restricted Kohn-Sham')

        if(nbf > nif):
            old_mf = dft.RKS(mol)
            mf = scf.remove_linear_dep_(old_mf, threshold=1.1e-6, lindep=1.1e-6)
        elif(nbf == nif):
            mf = dft.RKS(mol)
        else:
            raise ValueError('nbf<nif. This is impossible.')
        nocc = mol.nelectron // 2
        mf.mo_occ = np.asarray([2] * nocc + [0] * (nif - nocc))

    elif mo_coeff.ndim == 3:
        print('Unrestricted Kohn-Sham')
        if(nbf > nif):
            old_mf = dft.UKS(mol)
            mf = scf.remove_linear_dep_(old_mf, threshold=1.1e-6, lindep=1.1e-6)
        elif(nbf == nif):
            mf = dft.UKS(mol)
        else:
            raise ValueError('nbf<nif. This is impossible.')
        nocc_a, nocc_b = read_na_and_nb_from_fch(fch_file)
        mf.mo_occ = np.asarray([[1] * nocc_a + [0] * (nif - nocc_a),
                                [1] * nocc_b + [0] * (nif - nocc_b)])
    else:
        raise ValueError('Unknown dimension of mo_coeff: {}'.format(mo_coeff.ndim))
    mf.mo_coeff = mo_coeff
    mf.mo_energy = mo_energy
    mf.xc = functional
    if functional:
        print('functional: ', functional)
    return mf

def get_mf_from_molden(molden_file, functional, basis):
    '''

    molden_file: molden file generated by ORCA
    functional: functional used in ORCA
    basis: basis set name used in ORCA, 
    ORCA molden file does not contain unnoralized basis set information
    so we need to specify the basis set name and let pyscf to build the basis set
    '''
    from pyscf.tools import molden
    mol, mo_energy, mo_coeff, mo_occ, _, _ = molden.read(molden_file)

    mol.basis = basis
    mol.build(parse_arg = False)
    print('======= Molecular Coordinates ========')
    print(mol.atom)

    if isinstance(mo_coeff, tuple):
        mo_coeff = np.asarray(mo_coeff)
    if isinstance(mo_energy, tuple):
        mo_energy = np.asarray(mo_energy)

    if mo_coeff.ndim == 2:
        print('Restricted Kohn-Sham')
        mf = dft.RKS(mol)
    elif mo_coeff.ndim == 3:
        print('Unrestricted Kohn-Sham')
        mf = dft.UKS(mol)      
    mf.xc = functional
    mf.mo_occ = mo_occ
    mf.mo_coeff = mo_coeff
    mf.mo_energy = mo_energy

    return mf