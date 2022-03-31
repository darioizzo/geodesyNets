mamba env create -f environment.yml
mamba activate geodesynet
mamba install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install sobol_seq pyvista pyvistaqt
