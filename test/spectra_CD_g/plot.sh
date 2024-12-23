grep "Local Excited State" lvzilast_td_de.txt | awk '{print $5}' > lsqc_eV.txt
awk 'NR >1 {print $5}' lvzilast_td_de_1.fch-_TDDFT_ris_UV_spectra.txt > ris_CD.txt
awk 'NR >1 {print $6}' lvzilast_td_de_1.fch-_TDDFT_ris_UV_spectra.txt > ris_gfactor.txt


python spectra_CD_g.py --eV_file lsqc_eV.txt --CD_file  ris_CD.txt --filetypes lsqc-TDDFT-risp --outname lsqcCD --eV2nm_broaden True --nstates 11 --ylabel 'Rotatory Strength'

python spectra_CD_g.py --eV_file lsqc_eV.txt --g_file  ris_gfactor.txt --FWHM 0.1 --filetypes lsqc-TDDFT-risp --outname lsqcgfactor --eV2nm_broaden True --nstates 11 --ylabel 'g factor'