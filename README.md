# pyscf-ris
This repo demonstrates the semiempirical method TDDFT-ris.
A toy code for TDDFT-ris based on PySCF.

Also, see the TDDFT-ris+p plugin for Turbomole, [https://github.com/John-zzh/TDDFT-ris](https://github.com/John-zzh/TDDFT-ris)

## Quick run
You can directly fork&clone this repo and then
```
python test.py
```

It will perform TDDFT-ris calculation and then a standard TDDFT calculation. UV spectra file for both methods will be generated.

Feel free to test out larger molecules (20-99 atoms) listed in `xyz_files_EXTEST42` folder). You should expect an energy RMSE of 0.06 eV, and ~200x wall time speedup .

## Theory
1. Based on accurate KS ground state.
1. In the response part, use RIJK + minimal auxiliary basis (one s type Gaussian function per atom)
2. No exchang-correlation kernel.

## Reference
1. [Zhou, Z., Della Sala, F. and Parker, S.M., 2023. Minimal auxiliary basis set approach for the electronic excitation spectra of organic molecules. The Journal of Physical Chemistry Letters, 14, pp.1968-1976.](https://pubs.acs.org/doi/10.1021/acs.jpclett.2c03698)
2. [Giannone, G. and Della Sala, F., 2020. Minimal auxiliary basis set for time-dependent density functional theory and comparison with tight-binding approximations: Application to silver nanoparticles. The Journal of Chemical Physics, 153(8), p.084110.](https://doi.org/10.1063/5.0020545)
