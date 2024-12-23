
conda create -n mokit-py39 python=3.9
conda activate mokit-py39
conda install mokit pyscf -c mokit/label/cf -c conda-forge


# pip install matplotlib psutil

export PYTHONPATH=path_to/pyscf_TDDFT_ris:$PYTHONPATH