# pyscf-TDDFT-ris (v1.0)
This python package, based on PySCF, provides the semiempirical TDDFT-ris method, offering a quick and accurate calculation of TDDFT UV absorption spectra. The TDDFT-ris calculation starts from a completed DFT calculation, typically a `.fch` or `.molden` file. Also, it can also initiate from the PySCF `mf` object resulting from a DFT calculation.


Note: Turbomole has already built-in TDDFT-ris, see [the TDDFT-ris+p plugin for Turbomole](https://github.com/John-zzh/TDDFT-ris)

## Requirements
This project requires the following packages:
- python >= 3.8.0
- pyscf >= 2.1.0
- MOKIT (ORCA users can skip it)

### (1) install PysCF
[install PySCF](https://github.com/pyscf/pyscf) is pretty straightforward:
```
pip3 install pyscf
```
If any difficulty, please follow the [detailed PySCF installation guide](https://gitlab.com/jxzou/qcinstall/-/blob/main/%E7%A6%BB%E7%BA%BF%E5%AE%89%E8%A3%85PySCF-2.x.md).


### (2) install MOKIT
[MOKIT](https://gitlab.com/jxzou/mokit) is used to read the `.fch` file to initiate the TDDFT-ris calculation.

### (a) MacOS 
For MacOS users, install MOKIT with `homebrew` is recommended:
```
brew install mokit --HEAD --with-py38
```
You can change `py38` to any other python version that you can import PySCF, such as py310 
### (b) Linux 
For Linux users, download a pre-compiled MOKIT version is most convenient. After downloading the pre-built artifacts, you need to set the following environment
variables (assuming MOKIT is put in `$HOME/software/mokit`) in your `~/.bashrc`:
```
export MOKIT_ROOT=$HOME/software/mokit
export PATH=$MOKIT_ROOT/bin:$PATH
export PYTHONPATH=$MOKIT_ROOT:$PYTHONPATH
export LD_LIBRARY_PATH=$MOKIT_ROOT/mokit/lib:$LD_LIBRARY_PATH
export GMS=$HOME/software/gamess/rungms
```

If any difficulty, please follow the [detailed MOKIT installation guide](https://gitlab.com/jxzou/mokit), 4 options have been provided.


ORCA users can directly use the `.molden` file to initiate the TDDFT-ris calculation, and MOKIT is not required.

## Installation
First, clone this repository to your local machine:
```
git clone git@github.com:John-zzh/pyscf_TDDFT_ris.git
```
 
Then add the repo path to your `PYTHONPATH` environment by adding the following commands to your `~/.bash_profile` or `~/.bashrc` file:
```
export PYTHONPATH=path_to_your_dir:$PYTHONPATH
```
`path_to_your_directory` should be replaced with the path to the root directory, where you can see the `README.md` file. 

Adding the following `alias` to your `~/.bash_profile` or `~/.bashrc` file can save you some typing:
```
alias ris='python3.8 path_to_your_dir/main.py'
```
Again, change `python3.8` to any other python version that you can import PySCF.

## Run the calculation

The package has different interface for different software package, including Gaussian `.fch` file, ORCA `molden` file and PySCF `mf` object

### Gaussian software: `.fch` file
Suppose you have finished a DFT calculation with Gaussian and have a `.fch` file, you can run the TDDFT-ris calculation by executing the following command line:
```
ris -f your_fch_file -func pbe0
```
Optional Features
```
-f <filename>   
```
Choose a `.fch` input file that provides the basis set and molecular orbital coefficient information. For example, Gaussian `.fch` file or ORCA `molden` file. Default: *.fch

```
-func <functional_name>
```   
Specifies the functional name. Default: None

```
-b <basis>   
```
Specifies the basis set name. Only need to specify this option when dealing with a `molden` file. 


```
-ax <a_x>  
```
Specifies the amount of Fock exchange in the hybrid functional (e.g. -ax 0.25 for PBE0). This option only takes effect when the functional name is not given or the given functional name is not included in the library.  Default: None. 


```
-w <omega> 
```
Specifies the screening factor in the range-separated hybrid (RSH)functional (e.g. -w 0.3 for wb97x). This option only takes effect when the functional name is not given or the given functional name is not included in the library.  Default: None. 


```
-al <alpha>
```
Specifies the alpha factor in the RSH functional (e.g. -al 0.157706 for wb97x). This option only takes effect when the functional name is not given or the given functional name is not included in the library.  Default: None. 


```
-be <beta>
```
Specifies the alpha factor in the RSH functional (e.g. -be 0.842294 for wb97x). This option only takes effect when the functional name is not given or the given functional name is not included in the library.  Default: None. 

```
-th <theta>
```
Specifies the global parameter $\theta$ in the auxiliary basis $s$ orbital exponent, $\alpha_A = \frac{\theta}{R_A^2}$ Default: 0.2

```
-p <bool>
```
Specifies whether adding an extra $p$ function to the auxiliary basis. Default: False

```
-tda <bool>
```
Specifies whether using the TDA approximation. Default: False

```
-n <nroots>
```
Specifies the number of excited states to be calculated. Default: 20

```
-t <conv_tol>
```
Specifies the convergence tolerance for the Davidson algorithm. Default: 1e-5

```
-i <max_iter>
```
Specifies the maximum number of iterations for the Davidson algorithm. Default: 20







### ORCA software: `molden` file
Suppose you have finished a DFT calculation with ORCA and converted the `.gbw` file to `.molden` file, you can run the TDDFT-ris calculation by executing the following command line in directory containing the `.molden` file:
```
ris -f your_molden_file -func pbe0 -b def2-TZVP
```

### PySCF script: `mf` object


Suppose you use PySCF to build a `mf` object, you can run the TDDFT-ris calculation by constructing `TDDTF_ris` object. For example
```
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

    ''' also support UKS calculation '''
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

The calculation generates a `TDDFT-ris_UV_spectra.txt` file. I provided a script, `examples/Gaussian_fch/spectra.py`, to plot the spectra through command line `$sh plot.sh`.


Similar script to read ORCA `.molden` file is provided in the `/examples/ORCA_molden/` folder. Please note that MOKIT is not required in this way. Also, because `.molden` has normalized basis set, users need to manually input the basis set used in DFT calculation.

At present, this package exclusively supports **closed shell** systems with **hybrid** functional and range-separated hybrid (**RSH**) functional, as parameterized in `pyscf_TDDFT_ris/parameter.py`. If you require a functional that is not listed there, you can simply augment the dictionary with the name of the functional and its parameters. For instance, for a hybrid functional, you would add the Hartree-Fock (HF) component $a_x$, or for an RSH functional, you would include the $\omega$, $\alpha$, and $\beta$ values.



Feel free to test out larger molecules (20-99 atoms) in the `xyz_files_EXTEST42` folder. You should expect an energy RMSE of 0.06 eV, and ~300 wall time speedup compared to the standard TDDFT calculation.

## Bugs

I expect many bugs because I have not tested the robustness. Because I do not have Gaussian software to generate the `.fch` file :-)
I would appreciate it if any user provide error information.

## Theory
In the context of ab initio linear response TDDFT, we have introduced the TDDFT-ris model [1,2]. This is model achieved by two steps:
- approximate the two-electron repulsion integrals (ERI) using resolution-of-the-identity technique (RI) with only one $s$ type orbital per atom
- disable the exchange-correlation kernel. The exponents of the fitting basis are related to semiempirical atomic radii, which are available across the periodic table. Only one global parameter was adjusted across various hybrid exchange-correlation functional.

The TDDFT-ris model provides a nearly negligible deviation of just 0.06 eV for excitation energy calculations of organic molecules compared to traditional ab initio TDDFT. Moreover, it offers a significant computational advantage, being ~300 times faster with virtually no additional implementation effort. This represents a considerable improvement over the simplified TDDFT (sTDDFT) model, which shows a deviation of 0.24 eV.

Owing to its similar structure to ab initio TDDFT, the TDDFT-ris model can be readily integrated into most quantum chemistry packages, such as TURBOMOLE.

## To do list

1. Interface for other software packages that have not built-in the TDDFT-ris method, such as Qchem, NWChem, BDF.
2. Uniformed output file to support NTO analysis.
3. Assign certain elements with full default fitting basis, e.g. transition metals need full fitting basis
4. Solvation model

## Acknowledgements
Thank Dr. Zou for powerful MOKIT support. Thank gjj for the detailed guidance of MOKIT installaiton on MacOS system.

## Reference
To cite the TDDFT-ris method:
1. [Zhou, Z., Della Sala, F. and Parker, S.M., 2023. Minimal auxiliary basis set approach for the electronic excitation spectra of organic molecules. The Journal of Physical Chemistry Letters, 14, pp.1968-1976.](https://pubs.acs.org/doi/10.1021/acs.jpclett.2c03698)
2. [Giannone, G. and Della Sala, F., 2020. Minimal auxiliary basis set for time-dependent density functional theory and comparison with tight-binding approximations: Application to silver nanoparticles. The Journal of Chemical Physics, 153(8), p.084110.](https://doi.org/10.1063/5.0020545)
   
To cite the pyscf-TDDFT-ris package:
1. Zehao Zhou, pyscf-TDDFT-ris, https://github.com/John-zzh/pyscf_TDDFT_ris
2. Jingxiang Zou, Molecular Orbital Kit (MOKIT) https://gitlab.com/jxzou/mokit (see mored detailed citation instructions on MOKIT webpage)