
conda create -n mokit-py39 python=3.9
conda activate mokit-py39
conda install mokit pyscf -c mokit/label/cf -c conda-forge


# pip install matplotlib psutil

export PYTHONPATH=path_to/pyscf_TDDFT_ris:$PYTHONPATH




conda create -n mokit-cupy python=3.9
conda activate mokit-cupy 
pip install gpu4pyscf-cuda12x
pip install cutensor-cu12
conda install mokit -c mokit/label/cf -c conda-forge 
