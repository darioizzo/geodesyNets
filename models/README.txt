This folder contains:

1) the .mdl files for the best trained models (L1 loss) of all the homogeneous bodies considered. To load  them:
model = gravann.init_network(gravann.direct_encoding(), n_neurons=100, model_type="siren", activation = gravann.AbsLayer())

2) the .pk file containing the learned mascon cube model for the same bodies. To load them:
with open("mascons/name.pk", "rb") as file:
    mascon_points, mascon_masses, mascon_name = pk.load(file)

3) The csv produced by the validator on the mascon cubes