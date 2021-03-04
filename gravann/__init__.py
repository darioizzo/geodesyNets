"""
This module helps in the design and analysis of Artificial Neural Networks to represent the gravity field of celestial objects.
It was developed by the Advanced Conpcets team in the context of the project "ANNs for geodesy".
"""
import os

# Importing encodings for the spacial asteroid dimensions
from ._encodings import directional_encoding, positional_encoding, direct_encoding, spherical_coordinates

# Importing the losses
from ._losses import normalized_loss, mse_loss, normalized_L1_loss, contrastive_loss, normalized_sqrt_L1_loss, normalized_relative_L2_loss, normalized_relative_component_loss

# Importing the method to integrate the density rho(x,y,z) output of an ANN in the unit cube
from ._integration import ACC_ld, ACC_trap, U_mc, U_ld, U_trap_opt, rho_trap
from ._integration import sobol_points, compute_integration_grid

# Methods to load mascons etc.
from ._io import load_sample

# Importing misc methods for 3D graphics
from ._hulls import alpha_shape, ray_triangle_intersect, rays_triangle_intersect, is_outside, is_inside

# Importing the plots
from ._plots import plot_mascon, plot_model_grid, plot_model_rejection, plot_model_contours, plot_potential_contours
from ._plots import plot_mesh, plot_model_mesh, plot_point_cloud_mesh, plot_points, plot_model_mascon_acceleration
from ._plots import plot_model_vs_cloud_mesh, plot_gradients_per_layer, plot_model_vs_mascon_rejection, plot_model_vs_mascon_contours

# Importing the validation method
from ._validation import validation, validation_results_unpack_df, compute_c_for_model

# Importing methods to sample points around asteroid
from ._sample_observation_points import get_target_point_sampler

# Importing the mesh_conversion methods
from ._mesh_conversion import create_mesh_from_cloud, create_mesh_from_model

# Import the labeling functions the mascons
from ._mascon_labels import U_L, ACC_L, ACC_L_differential

# Import training utility functions
from ._train import init_network, train_on_batch, run_training, load_model_run

# Custom layer for siren
from .networks._abs_layer import AbsLayer

# Import utility functions
from ._utils import max_min_distance, enableCUDA, fixRandomSeeds, print_torch_mem_footprint, get_asteroid_bounding_box, EarlyStopping

# Set main device by default to cpu if no other choice was made before
if "TORCH_DEVICE" not in os.environ:
    os.environ["TORCH_DEVICE"] = 'cpu'
