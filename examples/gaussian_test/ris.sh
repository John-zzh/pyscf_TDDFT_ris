#!/bin/bash
# python3 ~/pyscf_TDDFT_ris/main.py -f methanol.fch -func pbe0 -n 5 -fout methanol -pt 0.05 -tda True
 

.~/software/stda1.6.3/stda_v1.6.3 -f methanol.molden
python3 ~/pyscf_TDDFT_ris/main.py -f methanol.fch -func pbe0 -fout methanol_pt2 -CSF True -tda True -n 5 -spectra False