# GeodesyNets
Code to train visualize and evaluate neural density fields using pytorch.

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
A geodesyNet represents the body density directly as a function of Cartesian coordinates. 
Recently, (see https://github.com/bmild/nerf)  a related architecture called Neural Radiance Fields (NeRF) was introduced to represent three-dimensional objects and complex scenes with an impressive accuracy learning from a set of two-dimensional images. The training of a NeRF solves the inverse problem of image rendering as it back-propagates the difference between images rendered from the network and a sparse set of observed images.

Similarly, the training of a geodesyNet solves the gravity inversion problem. The network learns from a dataset of measured gravitational accelerations back-propagating the difference to the corresponding accelerations computed from the density represented by the network.

The overall architecture to learn a neural density field is shown below:

![GeodesyNet Architecture](/figures/Fig1.png)

# Neural Density Field for 67p Churyumov-Gerasimenko
Once the network is trained we can explore and visualize the neural density field using techniques similar to 3D image scanning. This
results in videos such as the one below, obtained using the gravitational signature of the comet 67p Churyumov-Gerasimenko

![Neural Density Field for 67p](/figures/67p_low.gif)

# Neural Density Field for Bennu
Similarly, the video below refers to the results of differential training over a heterogenous Bennu model.

![Neural Density Field for 67p](/figures/bennu_low.gif)


