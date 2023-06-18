# pyscf-TDDFT-ris
This PySCF-based python package demonstrates the semiempirical method TDDFT-ris.
It can read `.fch` and `.molden` file and then run TDDFT-ris calculation.

Note: Turbomole has already built-in TDDFT-ris, see [the TDDFT-ris+p plugin for Turbomole](https://github.com/John-zzh/TDDFT-ris)

## Installation
This package requires pre-installation of [PySCF](https://github.com/pyscf/pyscf). `pip install pyscf` is just fine.

For Gaussian users, if you want to start calculation with `.fch` file, then you need to install [MOKIT](https://gitlab.com/jxzou/mokit). Use their pre-compiled version is most convenient.



You can directly `git clone` this repo and then put
```
export PYTHONPATH=path_to_your_dir:$PYTHONPATH
```
in your `~/.bash_profile` or `~/.bashrc` file to 'install' this package. `path_to_your_dir` is the path to the root directory where you can see `/pyscf_TDDFT_ris/` folder.




It will perform TDDFT-ris calculation and then a standard TDDFT calculation. UV spectra file for both methods will be generated.

Feel free to test out larger molecules (20-99 atoms) listed in `xyz_files_EXTEST42` folder). You should expect an energy RMSE of 0.06 eV, and ~200x wall time speedup .

## Theory
1. Based on accurate KS ground state.
1. In the response part, use RIJK + minimal auxiliary basis (one s type Gaussian function per atom)
2. No exchang-correlation kernel.

## Reference
1. [Zhou, Z., Della Sala, F. and Parker, S.M., 2023. Minimal auxiliary basis set approach for the electronic excitation spectra of organic molecules. The Journal of Physical Chemistry Letters, 14, pp.1968-1976.](https://pubs.acs.org/doi/10.1021/acs.jpclett.2c03698)
2. [Giannone, G. and Della Sala, F., 2020. Minimal auxiliary basis set for time-dependent density functional theory and comparison with tight-binding approximations: Application to silver nanoparticles. The Journal of Chemical Physics, 153(8), p.084110.](https://doi.org/10.1063/5.0020545)
