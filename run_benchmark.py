import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import toml

import gravann


def run(cfg, results_df):
    """This function runs all the permutations of above settings
    """
    print("#############  Initializing    ################")
    print("Using the following samples:", cfg["samples"])
    print("###############################################")
    gravann.enableCUDA()
    print("Will use device ", os.environ["TORCH_DEVICE"])
    print("###############################################")

    for sample in cfg["samples"]:
        print(f"\n--------------- STARTING {sample} ----------------")
        print(f"\nModel: {cfg['model']['type']}")
        # If limited, compute integration_domain, else will be [-1,1]^3
        if cfg["integration"]["limit_domain"]:
            cfg["integration"]["domain"] = gravann.get_asteroid_bounding_box(
                asteroid_pk_path="3dmeshes/" + sample)
        for loss in cfg["training"]["losses"]:
            for encoding in cfg["model"]["encoding"]:
                for batch_size in cfg["training"]["batch_sizes"]:
                    for target_sample_method in cfg["model"]["target_point_samplers"]:
                        for activation in cfg["model"]["activation"]:
                            print(
                                f"\n ---------- RUNNING CONFIG -------------")
                            print(
                                f"|LR={cfg['training']['lr']}\t\t\tloss={loss.__name__}\t\tencoding={encoding().name}|")
                            print(
                                f"|target_sample={target_sample_method}\tactivation={str(activation)[:-2]}\t\tbatch_size={batch_size}|")
                            print(
                                f"--------------------------------------------")
                            run_results = gravann.run_training(cfg, sample, loss,
                                                               encoding(), batch_size,
                                                               target_sample_method,
                                                               activation)
                            results_df = results_df.append(
                                run_results, ignore_index=True)
        print("###############################################")
        print("#############       SAMPLE DONE     ###########")
        print("###############################################")

    print(f"Writing results csv to {cfg['output_folder']}. \n")

    if os.path.isfile(cfg["output_folder"] + "/" + "results.csv"):
        previous_results = pd.read_csv(
            cfg["output_folder"] + "/" + "results.csv")
        results_df = pd.concat([previous_results, results_df])
    results_df.to_csv(cfg["output_folder"] + "/" + "results.csv", index=False)
    print("###############################################")
    print("#############   TUTTO FATTO :)    #############")
    print("###############################################")


def _init_env(cfg):
    # Select GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["cuda_devices"]

    # Select integrator and prepare folder path
    if cfg["model"]["use_acceleration"]:
        cfg["integrator"] = gravann.ACC_trap
        cfg["name"] = cfg["name"] + "_" + "ACC"
    else:
        cfg["integrator"] = gravann.U_trap_opt
        cfg["name"] = cfg["name"] + "_" + "U"

    cfg["name"] = cfg["name"] + "_" + cfg["model"]["type"]

    if cfg["training"]["visual_loss"]:
        cfg["name"] = cfg["name"] + "_" + "visual_loss"
    if cfg["integration"]["limit_domain"]:
        cfg["name"] = cfg["name"] + "_" + "limit_int"
    if cfg["training"]["differential_training"]:
        cfg["name"] = cfg["name"] + "_" + "diff_train"

    cfg["output_folder"] = "results/" + cfg["name"] + "/"

    # Init results dataframe
    results_df = pd.DataFrame(columns=["Sample", "Type", "Model", "Loss", "Encoding", "Integrator", "Activation",
                                       "Batch Size", "LR", "Target Sampler", "Integration Points", "Final Loss", "Final WeightedAvg Loss"])

    return cfg, results_df


def _cfg_to_func(cfg):
    losses, encodings, activations = [], [], []

    for loss in cfg["training"]["losses"]:
        losses.append(getattr(gravann, loss))

    for encoding in cfg["model"]["encoding"]:
        encodings.append(getattr(gravann, encoding))

    for activation in cfg["model"]["activation"]:
        if activation == "nn.Abs":
            activations.append(gravann.AbsLayer())
        else:
            activations.append(getattr(torch, encoding)())

    cfg["training"]["losses"] = losses
    cfg["model"]["encoding"] = encodings
    cfg["model"]["activation"] = activations
    return cfg


if __name__ == "__main__":
    cfg = toml.load(sys.argv[1])
    cfg = _cfg_to_func(cfg)
    cfg, results_df = _init_env(cfg)
    print(cfg)
    run(cfg, results_df)
