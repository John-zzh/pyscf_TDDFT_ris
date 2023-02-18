# pyscf-ris
TDDFT-ris for PySCF

A toy code for TDDFT-ris. 

You can directly pull this repo and 
```
python test.py
```

It will perform TDDFT-ris calculation and then a standard TDDFT calculation. UV spectra file for both methods will be generated.
Feel free to use larger molecules (20-99 atoms, in `xyz_files_EXTEST42` folder), and the expected energy errors are 0.06 eV.


##reference
[Zhou, Z., Della Sala, F. and Parker, S.M., 2023. Minimal auxiliary basis set approach for the electronic excitation spectra of organic molecules. The Journal of Physical Chemistry Letters, 14, pp.1968-1976.](https://pubs.acs.org/doi/10.1021/acs.jpclett.2c03698)


