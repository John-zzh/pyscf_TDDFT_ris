orca_2mkl methanol -mkl 
mkl2fch methanol.mkl methanol.fch
python3 ../../main.py -f methanol.fch -func pbe0 -M 100 -n 10  > ris.out