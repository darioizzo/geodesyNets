import pickle as pk
import torch
import gravann
import numpy as np
from copy import deepcopy
import os

if __name__ == "__main__":

    # If possible enable CUDA
    gravann.enableCUDA()
    gravann.fixRandomSeeds()
    device = os.environ["TORCH_DEVICE"]
    print("Will use device ", device)

    mascon_ground_truths_pickles = [
        "Eros.pk",
        "Bennu.pk",
        "Itokawa.pk",
        "Churyumov-Gerasimenko.pk",
        "Torus.pk",
        "Hollow2.pk"
    ]

    for filename in mascon_ground_truths_pickles:
        # Let us load the mascon ground truths
        with open("mascons/" + filename, "rb") as file:
            mascon_points, mascon_masses, mascon_name = pk.load(file)
        mascon_points = torch.tensor(mascon_points)
        mascon_masses = torch.tensor(mascon_masses)
        print("\nModel ground truth pickle: ", filename)
        print("Model ground truth name: ", mascon_name)
        name = filename.split(".")[0]

        # We build the masconCUBE
        N = 45
        # Here we define the sqrt(m_j) or model parameters
        mascon_cube_masses = torch.rand((N*N*N, 1))*2-1
        mascon_cube_masses = mascon_cube_masses.requires_grad_(True)

        # Here we define the points of the masconCUBE
        X, Y, Z = torch.meshgrid(
            torch.linspace(-1, 1, N), torch.linspace(-1, 1, N), torch.linspace(-1, 1, N), indexing='ij')
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        Z = Z.reshape(-1, 1)
        mascon_cube_points = torch.concat((X, Y, Z), dim=1)

        # Training params
        batch_size = 1000

        # Loss. The normalized L1 loss.
        loss_fn = gravann.normalized_L1_loss

        # In this case we sample points unifromly in a sphere and reject those that are inside the asteroid
        targets_point_sampler = gravann.get_target_point_sampler(
            batch_size,
            limit_shape_to_asteroid="3dmeshes/" + name + "_lp.pk",
            method="spherical",
            bounds=[0, 1])

        # Here we set the optimizer
        learning_rate = 1e-1
        optimizer = torch.optim.Adam(
            params=[mascon_cube_masses], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.8, patience=200, min_lr=1e-8, verbose=True)

        # And init the best results
        best_loss = np.inf

        # TRAINING LOOP (normal training, no use of any prior shape information)------------------------
        # This cell can be stopped and started again without loosing memory of the training nor its logs
        torch.cuda.empty_cache()
        # The main training loop
        for i in range(5000):
            # Each ten epochs we resample the target points
            if (i % 10 == 0):
                target_points = targets_point_sampler()
                # We compute the labels whenever the target points are changed
                labels = gravann.ACC_L(
                    target_points, mascon_points, mascon_masses)

            # We compute the values predicted by the neural density field
            predicted = gravann.ACC_L(
                target_points, mascon_cube_points, mascon_cube_masses*mascon_cube_masses)

            # We compute the scaling constant (k in the paper) used in the loss
            c = torch.sum(predicted*labels)/torch.sum(predicted*predicted)

            # We compute the loss
            loss = loss_fn(predicted.view(-1), labels.view(-1))

            # We store the model if it has the lowest fitness
            # (this is to avoid losing good results during a run that goes wild)
            if loss < best_loss:
                best_model = deepcopy(mascon_cube_masses)
                best_loss = loss
                print(".", end="", flush=True)

            # Print every i iterations
            if i % 250 == 0:
                print(" "+str(i)+" ", end="", flush=True)

            # Zeroes the gradient (necessary because of things)
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()

            # Perform a step in LR scheduler to update LR
            scheduler.step(loss.item())

        print("\nLoss is: ", best_loss)
        final_mascon_masses = best_model*best_model / \
            torch.sum(best_model*best_model)
        results = gravann.validation_mascon(mascon_cube_points,
                                            final_mascon_masses,
                                            mascon_points,
                                            mascon_masses,
                                            asteroid_pk_path="3dmeshes/" + filename,
                                            N=10000,
                                            batch_size=100,
                                            progressbar=True)
        # Write results for pablo
        results.to_csv(name + "_mascon_validation_results.csv", index=False)
        with open("mascons/"+name+"_masconCUBE.pk", "wb") as file:
            pk.dump((mascon_cube_points.cpu().numpy(), final_mascon_masses.view(-1).detach().cpu().numpy(), name + " masconCUBE"), file)
