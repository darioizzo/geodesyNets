# core stuff
import gravann
import os
import glob
import pandas as pd
import numpy as np

# name = "VALIDATION"
# root_folder = "results/validate/"
# differential_training = False
# russell_points = 3

# name = "VALIDATION_NU"
# root_folder = "results/validate_nu/"
# differential_training = False
# russell_points = 3

# name = "VALIDATION_EROS"
# root_folder = "results/validate_eros/"
# differential_training = False
# russell_points = 50

root_folder = "results/validate_differential_siren/"
differential_training = True
name = "VALIDATION_DIFF"
russell_points = 3

# If possible enable CUDA
gravann.enableCUDA()
gravann.fixRandomSeeds()
device = os.environ["TORCH_DEVICE"]
print("Will use device ", device)

# Pick sampling altitude
sampling_altitudes = np.asarray([0.05, 0.1, 0.25])
sample_altitude_constant = {
    "Bennu.pk": 0.5 * 0.5634379088878632 / 0.3521486930549145,
    "Bennu_nu.pk": 0.5 * 0.5634379088878632 / 0.3521486930549145,
    "Churyumov-Gerasimenko.pk": 0.5 * 5.0025703125 / 3.1266064453124995,
    "Eros.pk": 0.5 * 32.66218376159668 / 20.413864850997925,
    "Itokawa.pk": 0.5 * 0.5607019066810608 / 0.350438691675663,
    "Itokawa_nu.pk": 0.5 * 0.5607019066810608 / 0.350438691675663,
    "Hollow.pk": 0.8,
    "Hollow_nu.pk": 0.8,
    "Torus.pk": 0.8
}


# Find present models
root_folder = root_folder.replace("\\", "/")
model_files = glob.glob(root_folder + "/**/best_model.mdl", recursive=True)
folders = [model_file.split("best_model.mdl")[0] for model_file in model_files]
print("Found the following:")
[print(folder) for folder in folders]

results_df = pd.DataFrame()
for folder in folders:
    # Load run
    model, encoding, sample, c, use_acc, mascon_points, mascon_masses_u, mascon_masses_nu, cfg = gravann.load_model_run(
        folder, differential_training)

    # Validate
    validation_results = gravann.validation(
        model, encoding, mascon_points, mascon_masses_u, use_acc, "3dmeshes/"+sample,
        sampling_altitudes=sample_altitude_constant[sample] *
        sampling_altitudes,
        N_integration=500000,
        N=10000,
        russell_points=russell_points,
        mascon_masses_nu=mascon_masses_nu,
        c=c
    )

    # Compute validation results
    val_res = gravann.validation_results_unpack_df(validation_results)

    # Accumulate results
    result_dictionary = {"Sample": sample,
                         "Loss": cfg["Loss"],
                         "Altitudes": sample_altitude_constant[sample] * sampling_altitudes}

    run_results = pd.concat(
        [pd.DataFrame([result_dictionary]), val_res], axis=1)
    results_df = results_df.append(run_results, ignore_index=True)

results_df.to_csv("results/" + name + ".csv")  # store as csv
