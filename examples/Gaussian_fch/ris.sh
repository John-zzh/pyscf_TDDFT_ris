
# python3 ../../main.py -f 2periacene_CAM-B3LYP.fch -func cam-b3lyp -fout WTFFFFF -pt 0.05 -tda True -n 5

# python3 ../../main.py -f 6-31+_112.fch -func m06-2x -fout 6-31+_112_davidson -pt 0.1 -CSF False  -tda True -n 10 > standard_ris.txt
python3 ../../main.py -f 6-31+_112.fch -func m06-2x -fout 6-31+_112_CSF -pt 0.1 -CSF True  -tda True -n 10 -pt2_tol 1e-4 -N 10 -single True -specw 10 -truncMO False -spectra True -approx ris
