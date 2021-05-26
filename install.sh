conda env create -f environment.yml
conda activate geodesynet
conda install pytorch=1.6.0 cudatoolkit -c pytorch
pip install sobol_seq pyvista pyvistaqt