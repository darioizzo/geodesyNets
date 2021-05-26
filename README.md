# GeodesyNets
Code to train visualize and evaluate neural density fields.

## Installation

We recommend using a [conda](https://docs.conda.io/en/latest/) environment to run this code. Once you have conda, you can simply execute the `install.sh` script to create a conda environment called `geodesynet` with all required modules. In case you want to install in another way or encounter problems, you need to install the following depedencies:

```
python=3.8
ipython
scikit-learn
numpy
matplotlib
jupyter
tqdm>=4.50.0
pandas 
pytorch=1.6.0 
cudatoolkit # if you want to utilize GPUs
sobol_seq  # only on pypi
pyvista # only on pypi
pyvistaqt # only on pypi
```

# Architecture at a glance
![GeodesyNet Architecture](/figures/Fig1.png)

# Neural Density Field for 67p Churyumov-Gerasimenko
![Neural Density Field for 67p](/figures/67p_low.gif)

