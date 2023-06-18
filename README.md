# pyscf-TDDFT-ris (Beta 1.0)
This python package, based on PySCF, provides the semiempirical TDDFT-ris method, offering a quick and precise calculation of TDDFT UV absorption spectra. The TDDFT-ris computation begins from a completed DFT calculation, typically indicated by a `.fch` or `.molden` file. Also, it can also initiate from the PySCF `mf` object resulting from a DFT calculation.


Note: Turbomole has already built-in TDDFT-ris, see [the TDDFT-ris+p plugin for Turbomole](https://github.com/John-zzh/TDDFT-ris)

## Requirements
This package requires `>=python3.7.0` and pre-installation of [PySCF](https://github.com/pyscf/pyscf). `pip install pyscf` is just fine.

For Gaussian users, you want to feed the pyscf-TDDFT-ris package with `.fch` file, and you need to install [MOKIT](https://gitlab.com/jxzou/mokit), which is used to read the `.fch` file. Use the pre-compiled MOKIT is most convenient.


It's quite easy to install PySCF and MOKIT.

## Installation
You can clone this repository using the git clone command. After that, you need to make the package importable by adding it to your `PYTHONPATH`. To do this, you can add the following line to your `~/.bash_profile` or `~/.bashrc` file:
```
export PYTHONPATH=path_to_your_dir:$PYTHONPATH
```
In the above command, path_to_your_directory should be replaced with the path to the root directory where you can see the `/pyscf_TDDFT_ris/` folder.

## Run the calculation

The package follows the PySCF style to run the TDDFT-ris calculation. Take `examples/Gaussain_fch/with_fch.py` as an example,

```
from pyscf_TDDFT_ris import TDDFT_ris

mol, mf = TDDFT_ris.get_mol_mf_fch('2periacene_CAM-B3LYP.fch', 'CAM-B3LYP')

td = TDDFT_ris.TDDFT_ris(mf, mol, add_p=False, nroots = 20, max_iter=30)
energies, X, Y = td.kernel_TDDFT()
print('Excitations energies:')
print(energies)
print('==================')
```

The function `get_mol_mf` extracts **molecular coordinates**, **basis set**, **KS orbital energies** and **coefficients matrix** from the `2periacene_CAM-B3LYP.fch` file. You need to manually input the functional name you used in your DFT calculation. In this case it uses `CAM-B3LYP` functional. `mol` and `mf` objects contain all these ground state information, and they are the input of the TDDFT-ris diagonalization precedure. `add_p=False` means only include one `s` type orbital in the axillary basis; `add_p=True` means adding one more `p` orbital for non-Hydrogen atoms. Adding `p` orbital can improve the accuracy, but also slows down the calculation by ~2.7 folds (for organic systems).

Navigate to the directory containing the `.fch` file and execute:
```
python with_fch.py
```

Similar script to read ORCA `.molden` file is provided in the `/examples/ORCA_molden/` folder. Please note that MOKIT is not required in this way. Also, because `.molden` has normalized basis set, users need to manually input the basis set used in DFT calculation.

At present, this package exclusively supports **closed shell** systems with **hybrid** functional and range-separated hybrid (**RSH**) functional, as parameterized in `pyscf_***REMOVED***`. If you require a functional that is not listed there, you can simply augment the dictionary with the name of the functional and its parameters. For instance, for a hybrid functional, you would add the Hartree-Fock (HF) component $a_x$, or for an RSH functional, you would include the $\omega$, $\alpha$, and $\beta$ values.

The calculation generates a `***REMOVED***` file. I provided a script, `examples/Gaussian_fch/spectra.py`, to plot the spectra through command line `$sh plot.sh`.

Feel free to test out larger molecules (20-99 atoms) in the `xyz_files_EXTEST42` folder. You should expect an energy RMSE of 0.06 eV, and ~300 wall time speedup compared to the standard TDDFT calculation.

## Bugs

I expect many bugs because I have not tested the robustness. Because I do not have the software to generate the `.fch` file :-)
I would appreciate it if any user provide error information.

## Theory
In the context of ab initio hybrid TDDFT with the resolution-of-the-identity (RI) approach, we have introduced the TDDFT-ris model [1,2]. This is achieved by approximating two-electron integrals using a minimal auxiliary basis and disabling the exchange-correlation kernel. The exponents of the fitting basis are related to semiempirical atomic radii, which are available across the periodic table. Only one global parameter was adjusted across various hybrid exchange-correlation functional.

The TDDFT-ris model provides a nearly negligible deviation of just 0.06 eV for excitation energy calculations of organic molecules compared to traditional ab initio TDDFT. Moreover, it offers a significant computational advantage, being ~300 times faster with virtually no additional implementation effort. This represents a considerable improvement over the simplified TDDFT (sTDDFT) model, which shows a deviation of 0.24 eV.

Owing to its similar structure to ab initio TDDFT, the TDDFT-ris model can be readily integrated into most quantum chemistry packages, such as TURBOMOLE.

## To do list
1. Open shell system (UKS).
2. Pure functional (actually, it currently can, just add a functional with $a_x =0$ in the `parameter.py` file.)
3. Interface for other software packages that have not built-in the TDDFT-ris method, such as Qchem, NWChem, BDF.
4. Uniformed output file to support NTO analysis.
5. Assign certain elements with full default fitting basis, e.g. transition metals with `d` orbitals
6. Solvation model

## Acknowledgements
Thank Dr. Zou for powerful MOKIT support. Thank gjj for the detailed guidance of MOKIT installaiton on MacOS system.

## Reference
To cite the TDDFT-ris method:
1. [Zhou, Z., Della Sala, F. and Parker, S.M., 2023. Minimal auxiliary basis set approach for the electronic excitation spectra of organic molecules. The Journal of Physical Chemistry Letters, 14, pp.1968-1976.](https://pubs.acs.org/doi/10.1021/acs.jpclett.2c03698)
2. [Giannone, G. and Della Sala, F., 2020. Minimal auxiliary basis set for time-dependent density functional theory and comparison with tight-binding approximations: Application to silver nanoparticles. The Journal of Chemical Physics, 153(8), p.084110.](https://doi.org/10.1063/5.0020545)
   
To cite the pyscf-TDDFT-ris package:
1. Zehao Zhou, pyscf-TDDFT-ris, https://github.com/John-zzh/pyscf_TDDFT_ris
2. Jingxiang Zou, Molecular Orbital Kit (MOKIT) https://gitlab.com/jxzou/mokit (see mored detailed citation instructions on MOKIT webpage)