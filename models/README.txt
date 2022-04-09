This folder contains:

1) the .mdl files for the best trained models (L1 loss) of all the homogeneous and heterogeneous (_nu) bodies considered (corresponding to the zenodo best_model.mdl).
To load  them:
model = gravann.init_network(gravann.direct_encoding(), n_neurons=100, model_type="siren", activation = gravann.AbsLayer())

2) the .pk file containing the learned mascon cube model for same of the bodies. To load them:
with open("mascons/name.pk", "rb") as file:
    mascon_points, mascon_masses, mascon_name = pk.load(file)