# Import our module containing helper functions
import gravann

# Core imports
import numpy as np
import pickle as pk
import os
from collections import deque

# pytorch
from torch import nn
import torch

# plotting stuff
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import argparse
import pickle


parser = argparse.ArgumentParser(
    prog="direct_training",
    description="Train geodesynet",
)
parser.add_argument(
    "-a", "--asteroid", required=True, type=str,
    choices=[
        "eros", "bennu", "itokawa", "churyumov-gerasimenko", "planetesimal",
        "torus", "bennu_nu", "itokawa_nu", "planetesimal_nu"
    ]
)
parser.add_argument("-s", "--save-name", required=False, type=str, default="classic")
parser.add_argument("-n", "--n-neurons", default=100, type=int)
parser.add_argument("-l", "--n-hidden-layers", default=9, type=int)
parser.add_argument("-m", "--model-type", default="siren", type=str, choices=["siren", "siren_q"])
parser.add_argument(
    "-q", "--n-quadrature", default=30000, type=int,
    help="Number of points to be used to evaluate numerically the triple integral defining the acceleration."
)
parser.add_argument(
    "-b", "--batch-size", default=100, type=int,
    help="Dimension of the batch size, i.e. number of points where the ground truth is compared to the predicted acceleration at each training epoch."
)


args = parser.parse_args()
name_of_gt = args.asteroid

# If possible enable CUDA
gravann.enableCUDA()
gravann.fixRandomSeeds()
device = os.environ["TORCH_DEVICE"]
print("Will use device ",device)

# We load the ground truth (a mascon model of some body)
with open("mascons/"+name_of_gt+".pk", "rb") as file:
    mascon_points, mascon_masses, mascon_name = pk.load(file)
    
mascon_points = torch.tensor(mascon_points)
mascon_masses = torch.tensor(mascon_masses)

# Print some information on the loaded ground truth 
# (non-dimensional units assumed. All mascon coordinates are thus in -1,1 and the mass is 1)
print("Name: ", mascon_name)
print("Number of mascons: ", len(mascon_points))
print("Total mass: ", sum(mascon_masses))

# Encoding: direct encoding (i.e. feeding the network directly with the Cartesian coordinates in the unit hypercube)
# was found to work well in most cases. But more options are implemented in the module.
encoding = gravann.direct_encoding()

# The model is here a SIREN network (FFNN with sin non linearities and a final absolute value to predict the density)
model = gravann.init_network(
    encoding, n_neurons=args.n_neurons, model_type=args.model_type,
    activation = gravann.AbsLayer(), hidden_layers=args.n_hidden_layers
)

# When a new network is created we init empty training logs
loss_log = []
weighted_average_log = []
running_loss_log = []
n_inferences = []
# .. and we init a loss trend indicators
weighted_average = deque([], maxlen=20)

# Once a model is loaded the learned constant c (named kappa in the paper) is unknown 
# and must be relearned (ideally it should also be saved at the end of the training as it is a learned parameter)
c = gravann.compute_c_for_model(model, encoding, mascon_points, mascon_masses, use_acc = True)


# Training of a geodesyNet

n_quadrature = args.n_quadrature
batch_size = args.batch_size

# Loss. The normalized L1 loss (kMAE in the paper) was
# found to be one of the best performing choices.
# More are implemented in the module
loss_fn = gravann.normalized_L1_loss

# The numerical Integration method. 
# Trapezoidal integration is here set over a dataset containing acceleration values,
# (it is possible to also train on values of the gravity potential, results are similar)
mc_method = gravann.ACC_trap

# The sampling method to decide what points to consider in each batch.
# In this case we sample points unifromly in a sphere and reject those that are inside the asteroid
targets_point_sampler = gravann.get_target_point_sampler(batch_size, 
                                                         limit_shape_to_asteroid="3dmeshes/"+name_of_gt+"_lp.pk", 
                                                         method="spherical", 
                                                         bounds=[0,1])
# Here we set the optimizer
learning_rate = 1e-4
optimizer = torch.optim.Adam(
    [
        {'params': [p for n, p in model.named_parameters() if not gravann.is_quadratic_param(n)]},
        {'params': [p for n, p in model.named_parameters() if gravann.is_quadratic_param(n)], 'lr': learning_rate/10},
    ],
    lr=learning_rate
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor = 0.8, patience = 200, min_lr = 1e-6,verbose=True)

# And init the best results
best_loss = np.inf
best_model_state_dict = model.state_dict()

# TRAINING LOOP (normal training, no use of any prior shape information)------------------------
torch.cuda.empty_cache()
# The main training loop
for i in range(20000):
    # Each ten epochs we resample the target points
    if (i % 10 == 0):
        target_points = targets_point_sampler()
        # We compute the labels whenever the target points are changed
        labels = gravann.ACC_L(target_points, mascon_points, mascon_masses)
    
    # We compute the values predicted by the neural density field
    predicted = mc_method(target_points, model, encoding, N=n_quadrature, noise=0.)
    
    # We learn the scaling constant (k in the paper)
    c = torch.sum(predicted*labels)/torch.sum(predicted*predicted)
    
    # We compute the loss (note that the contrastive loss needs a different shape for the labels)
    if loss_fn == gravann.contrastive_loss:
       loss = loss_fn(predicted, labels)
    else:
       loss = loss_fn(predicted.view(-1), labels.view(-1))
    
    # We store the model if it has the lowest fitness 
    # (this is to avoid losing good results during a run that goes wild)
    if loss < best_loss:
        best_model_state_dict = model.state_dict()
        best_loss = loss
        print('New Best: ', loss.item())
        # Uncomment to save the model during training (careful it overwrites the model folder)
        #torch.save(model.state_dict(), "models/"+name_of_gt+".mdl")
    
    # Update the loss trend indicators
    weighted_average.append(loss.item())
    
    # Update the logs
    weighted_average_log.append(np.mean(weighted_average))
    loss_log.append(loss.item())
    n_inferences.append((n_quadrature*batch_size) // 1000000) #counted in millions
    
    # Print every i iterations
    if i % 25 == 0:
        wa_out = np.mean(weighted_average)
        print(f"It={i}\t loss={loss.item():.3e}\t  weighted_average={wa_out:.3e}\t  c={c:.3e}")
        
    # Zeroes the gradient (necessary because of things)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
    
    # Perform a step in LR scheduler to update LR
    scheduler.step(loss.item())

# Here we restore the learned parameters of the best model of the run
for layer in model.state_dict():
    model.state_dict()[layer] = best_model_state_dict[layer]


name_of_model = args.save_name
torch.save(model.state_dict(), "models/"+name_of_gt+"__"+name_of_model+".mdl")

log_dict = {"loss": loss_log, "weighted_average_loss": weighted_average_log}
with open("logs/"+name_of_gt+"__"+name_of_model+".pkl", 'wb') as f:
    pickle.dump(log_dict, f)
