**Read this in other languages: [English](README.md), [中文](README_zh.md).**


# TDDFT-ris (v2.0)
This python package, based on PySCF, provides the semiempirical TDDFT-ris method, offering a quick and accurate calculation of TDDFT UV-vis absorption spectra. The TDDFT-ris calculation starts from a completed SCF calculation, in Gaussian`.fch` file format. Outputs from other programs can be converted to `.fch` file easily through MOKIT.Also, it can start from the PySCF `mf` object.

Currently, it supports RKS-TDDFT, hybrid/range-separated-hybrid functional. Not yet support implicit solvation method.

Note: 

(1) Software package TURBOMOLE7.7dev has already built-in TDDFT-ris, see [the TDDFT-ris+p plugin for Turbomole](https://github.com/John-zzh/TDDFT-ris)

(2) Software package Amespv2.1dev has already built-in TDDFT-ris, see [Amesp](https://amesp.xyz/)

## Theory
In the context of ab initio linear response TDDFT, we have introduced the TDDFT-ris method [1,2]. This is method achieved by two steps:
- approximate the two-electron integrals using resolution-of-the-identity technique (**RI**) with only one **$s$** type orbital per atom
- disable the exchange-correlation kernel. 

The exponents $\alpha_A$ of the **$s$** type orbital centered on atom $A$ is related to the tabulated semi-empirical atomic radii $R_A$. Only one global parameter $\theta$ was fine-tuned across various hybrid exchange-correlation functional.

$\alpha_A = \frac{\theta}{R_A^2}$

Compared to traditional ab initio TDDFT, for excitation energy calculations of organic molecules, the TDDFT-ris method provides a nearly negligible deviation of just 0.06 eV. Moreover, it offers a significant computational advantage, being ~300 times faster. This represents a considerable improvement over the [simplified TDDFT (sTDDFT) method](https://github.com/grimme-lab/stda), which shows an energy deviation of 0.24 eV.

Owing to the similar structure to ab initio TDDFT, the TDDFT-ris method can be readily integrated into most quantum chemistry packages with virtually no additional implementation effort. Software packages such as [TURBOMOLE7.7dev](https://www.turbomole.org/turbomole/release-notes-turbomole-7-7/) and [Amespv1.1dev](https://amesp.xyz/) have already built-in the TDDFT-ris method.

[ORCA6.0](https://github.com/ORCAQuantumChemistry/CompoundScripts) supports TDDFT-ris calculation through compound script.

However, if you need high performance, use my code :-)

## Install TDDFT-ris-pyscf
First, clone this repository to your local machine:
```
git clone git@github.com:John-zzh/TDDFT-ris-pyscf.git
```
 
Then add the repo path to your `PYTHONPATH` environment by adding the following commands to your `~/.bash_profile` or `~/.bashrc` file:
```
export PYTHONPATH=absolue_path_to_ris_repo:$PYTHONPATH
```
`absolue_path_to_ris_repo` should be replaced with the absolute path to the cloned repository directory, where you can see the `README.md` file. 



### install MOKIT
[MOKIT](https://gitlab.com/jxzou/mokit) is used to read the `.fch` file to read the ground state information. Install MOKIT with these commands:

```
conda create -n ris-mokit-pyscf-py39 python=3.9
conda activate ris-mokit-pyscf-py39
conda install mokit pyscf -c mokit/label/cf -c conda-forge
```

Alternatively, see other install options at [MOKIT installation guide](https://gitlab.com/jxzou/mokit). The MOKIT developer Dr. Zou is very active. 



## Run the calculation

The calculation requires a ground state calculation output, either Gaussian `.fch` file or PySCF `mf` object

### Gaussian software:`.fch` file
Suppose you have finished a PBE0 DFT calculation with Gaussian and have a `molecule.chk` file, you can run the TDDFT-ris calculation for 10 lowest excited states on 4 CPUs by executing the following commands:
```
formchk molecule.chk molecule.fch
conda activate ris-mokit-pyscf-py39
export PYTHONPATH=absolue_path_to_ris_repo:$PYTHONPATH
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
python absolue_path_to_ris_repo/main.py -f molecule.fch -func pbe0 -n 10 -M 8000
```
NumPy provides major parallelization. 

## Command-Line Arguments

The following arguments can be passed to the program via the command line:

| **Argument**         | **Type**    | **Default**    | **Description**                                                                |
|----------------------|-------------|----------------|--------------------------------------------------------------------------------|
| `-f`                 | `str`       | `None`         | Input `.fch` filename (e.g., `molecule.fch`).                                  |
| `-fout`              | `str`       | `None`         | Output file name for spectra results.                                          |
| `-func`              | `str`       | `None`         | Functional name (e.g., `pbe0`).                                                |
| `-ax`                | `float`     | `None`         | HF component in the hybrid functional. Needed when '-func' is not provided     |
| `-w`                 | `float`     | `None`         | Screening factor in the range-separated functional.                            |
| `-alpha`             | `float`     | `None`         | Alpha parameter in the range-separated functional.                             |
| `-beta`              | `float`     | `None`         | Beta parameter in the range-separated functional.                              |
| `-theta`             | `int`       | `0.2`          | Exponent parameter: `theta / R^2`, with an optimal value of `0.2`.             |
| `-J_fit`             | `str`       | `sp`           | J fitting basis, options: `s`, `sp`, `spd`.                                    |
| `-K_fit`             | `str`       | `s`            | K fitting basis, options: `s`, `sp`, `spd`.                                    |
| `-M`                 | `int`       | `8000`         | Maximum memory usage in MB.                                                    |
| `-Ktrunc`            | `float`     | `40`           | Truncation threshold for MO in K, in eV.                                       |
| `-TDA`               | `bool`      | `False`        | Perform TDA calculation instead of TDDFT. **Deprecated for now**               |
| `-n`                 | `int`       | `10`           | Number of excited states to solve.                                             |
| `-t`                 | `float`     | `1e-3`         | Convergence tolerance in the Davidson diagonalization.                         |
| `-GS`                | `bool`      | `False`        | Use Gram-Schmidt orthogonalization. Default uses non-orthogonalized Krylov subspace (nKs) method. |
| `-i`                 | `int`       | `20`           | Maximum number of iterations in the Davidson diagonalization.                  |
| `-pt`                | `float`     | `0.05`         | Threshold for printing the transition coefficients.                            |
| `-spectra`           | `bool`      | `True`         | Print out the spectra file.                                                    |
| `-specw`             | `float`     | `10.0`         | Spectra window (in eV).                                                        |
| `-single`            | `bool`      | `True`         | Use single precision for calculations.                                         |



### Calcuate with PySCF `mf` object


Suppose you use PySCF to build a `mf` object, you can run the TDDFT-ris calculation by constructing `TDDTF_ris` object. For example
```
    import numpy as np
    from pyscf import gto,  dft
    from pyscf_TDDFT_ris import TDDFT_ris

    mol = gto.Mole()
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
    mol.build()

    ''' Run the SCF calculation. 
        Note: TDDFT-ris also supports UKS calculation 
    '''
    mf = dft.RKS(mol)

    mf = mf.density_fit() #optional 
    mf.conv_tol = 1e-10
    mf.grids.level = 3
    mf.xc = 'pbe0'
    mf.kernel()

    ''' build an TDDFT_ris orbject '''
    td = TDDFT_ris.TDDFT_ris(mf, nroots=20)
            
    ''' TDDFT-ris calculation '''
    energies, X, Y, oscillator_strength = td.kernel_TDDFT() 

    ''' TDA-ris calculation '''
    energies, X, oscillator_strength = td.kernel_TDA() 
```

The calculation generates a `***REMOVED***` file. I provided a script, `examples/Gaussian_fch/spectra.py`, to plot the spectra through command line `$sh plot.sh`.


Feel free to test out larger molecules (20-99 atoms) in the `xyz_files_EXTEST42` folder. You should expect an energy RMSE of 0.06 eV, and ~1000 wall time speedup compared to the standard TDDFT calculation.

## Bugs
TDA codes are not opimized yet, do not use it. Anyway, TDDFT-ris is better for UV spectra.

## To do list
1. Assign certain elements with full default fitting basis, e.g. transition metals need full fitting basis
2. TDA, UKS, pure functionals.
3. Solvation model (not important for now)

## Acknowledgements
Thank Dr. Zou (the developer of MOKIT) for powerful interface support. Thank gjj for the detailed guidance of MOKIT installaiton on MacOS system. Thank Dr. Zhang (the developer of Amesp) for cross validation with TDDFT-ris implementation. Thank Dr. Della Sala for the development of TDDFT-as prototype, and contribution to the development of the TDDFT-ris method [1, 2].

## Reference
To cite the TDDFT-ris method:
1. [Zhou, Zehao, Fabio Della Sala, and Shane M. Parker. 
     Minimal auxiliary basis set approach for the electronic excitation spectra 
    of organic molecules. The Journal of Physical Chemistry Letters 
    14, no. 7 (2023): 1968-1976.](https://pubs.acs.org/doi/10.1021/acs.jpclett.2c03698)(must cite)
2. [Zhou, Zehao, and Shane M. Parker. 
            Converging Time-Dependent Density Functional Theory Calculations in Five Iterations 
            with Minimal Auxiliary Preconditioning. Journal of Chemical Theory and Computation 
            20, no. 15 (2024): 6738-6746.](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00577) 
            (for efficient orbital truncation technique)
3. [ Zhou, Zehao, and Shane M. Parker. 
            Converging Time-Dependent Density Functional Theory Calculations in Five Iterations 
            with Minimal Auxiliary Preconditioning. Journal of Chemical Theory and Computation 
            20, no. 15 (2024): 6738-6746. ](https://doi.org/10.1063/5.0020545)(The idea of TDDFT-ris originates from TDDFT-as)
   
To cite the pyscf-TDDFT-ris package:
1. Zehao Zhou, pyscf-TDDFT-ris, https://github.com/John-zzh/pyscf_TDDFT_ris
2. Jingxiang Zou, Molecular Orbital Kit (MOKIT) https://gitlab.com/jxzou/mokit (see mored detailed citation instructions on MOKIT webpage)
3. PySCF: the Python-based simulations of chemistry framework, 
           Q. Sun, et. al., and G. K.-L. Chan, WIREs Comput. Mol. Sci. 8, e1340 (2018) 
           (https://pyscf.org/about.html)